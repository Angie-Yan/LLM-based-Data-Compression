import os
import json
import time
import queue
import threading
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel

# ==========================================
# Medusa Head Structure (need to be as same as training)
# ==========================================

class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act     = nn.SiLU()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear(hidden_states))
        x = x + hidden_states
        return self.lm_head(x)


class MedusaHeadCollection(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size) for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor, num_heads: int = None):
        """
        num_heads: 只跑前 num_heads 个 head 的前向。
        默认为 None 时使用全部训练好的 head 数（向后兼容旧调用方式）。
        这样当你只想用前 n 个 head 压缩时，不会浪费算力去跑没用到的 head。
        """
        n = self.num_heads if num_heads is None else num_heads
        if not (1 <= n <= self.num_heads):
            raise ValueError(
                f"num_heads={n} 超出范围，训练好的 head 总数是 {self.num_heads}"
            )
        return [head(hidden_states) for head in list(self.heads)[:n]]

    @classmethod
    def load(cls, save_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
        with open(os.path.join(save_dir, "medusa_config.json")) as f:
            config = json.load(f)
        heads = cls(config["hidden_size"], config["vocab_size"], config["num_heads"])
        state = torch.load(
            os.path.join(save_dir, "medusa_heads.pt"), map_location=device
        )
        heads.load_state_dict(state)
        # 保持和主模型一致的精度，否则 hidden_states 和 head 权重 dtype 不一致会报错
        return heads.to(device=device, dtype=dtype)


# ==========================================
# KV Cache 工具
# ==========================================
def slice_kv_cache(past_key_values, window_length):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i]   = past_key_values.key_cache[i][..., -window_length:, :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][..., -window_length:, :]
        return past_key_values
    return tuple(
        (k[:, :, -window_length:, :], v[:, :, -window_length:, :])
        for k, v in past_key_values
    )


def get_kv_len(past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].size(2)


# ==========================================
# GPU显存 / 精度 工具
# ==========================================

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _is_cuda(device) -> bool:
    return "cuda" in str(device)


def warmup_and_reset_peak_memory(model, medusa_heads, input_ids, device, num_heads_to_run):
    """
    在正式计时前做一次“陪跑”前向，触发 cuDNN autotune / allocator 的一次性开销，
    然后立刻 reset_peak_memory_stats，保证后面测到的 peak 只反映真实压缩/解压阶段的显存占用，
    而不是被首次 kernel 选择、显存池首次扩容这些一次性成本污染。
    """
    if not _is_cuda(device):
        return
    torch.cuda.synchronize()
    with torch.inference_mode():
        B = input_ids.size(0)
        dummy_pos = torch.zeros((B, 1), dtype=torch.long, device=device)
        out = model(
            input_ids=input_ids,
            position_ids=dummy_pos,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden_states = out.hidden_states[-1][:, -1, :]
        _ = medusa_heads(hidden_states.unsqueeze(1), num_heads=num_heads_to_run)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_reserved_gb(device):
    if not _is_cuda(device):
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


# ==========================================
# Encoder
# ==========================================

class MedusaRankEncoder:
    """
    一次 forward 输出 num_active_heads 个位置的 logits：
      - medusa_logits[k]，k in [0, num_active_heads)

    每个位置独立算 rank，存成和原来一样的 "1" / "2,r" / "3,r" / "4,id" 格式，
    但现在每 step 写 (1+num_active_heads) 行 instead of 1 行。

    num_active_heads 可以 <= 训练时的 num_heads，不需要重新训练模型，
    只是压缩时选择用前几个 head 参与预测（n 越大，单次 forward 覆盖的 token 越多，
    但越靠后的 head 通常命中率越低）。
    """

    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model        = model
        self.medusa_heads = medusa_heads
        self.tokenizer     = tokenizer
        self.device         = device
        self.args           = args
        self.num_heads       = medusa_heads.num_heads          # 训练好的 head 总数
        self.num_active_heads = args.num_active_heads           # 本次实际参与压缩的 head 数
        self.step_size       = self.num_active_heads            # 每次 forward 推进的 token 数
        self.write_queue     = queue.Queue(maxsize=200)
        self.final_counts    = {1: 0, 2: 0, 3: 0, 4: 0}
        self.peak_vram_gb    = None   # encode 阶段 peak reserved 显存（GB）

    def _writer_worker(self, output_path, batch_size, num_real_steps, first_tokens):
        """
        Header 多存一个 step_size，decode 端需要知道每次 forward 推进几个 token
        (step_size 同时就是本次压缩实际用到的 head 数量，decode 端据此自动对齐)。
        每个 "逻辑 step"（即一次 forward）写 step_size 行，每行是该 batch 在该位置的 rank 信息。
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{batch_size} {num_real_steps} {self.args.window_length} {self.step_size}\n")
            f.write(" ".join(map(str, first_tokens)) + "\n")

            while True:
                item = self.write_queue.get()
                if item is None:
                    break

                # item: list of (ranks_cpu, targets_cpu)，长度 = step_size
                for ranks_cpu, targets_cpu in item:
                    step_data = []
                    for b in range(batch_size):
                        r = int(ranks_cpu[b])
                        if r == 1:
                            step_data.append("1")
                            self.final_counts[1] += 1
                        elif 2 <= r <= 4:
                            step_data.append(f"2,{r}")
                            self.final_counts[2] += 1
                        elif 5 <= r <= 32:
                            step_data.append(f"3,{r}")
                            self.final_counts[3] += 1
                        else:
                            step_data.append(f"4,{int(targets_cpu[b])}")
                            self.final_counts[4] += 1
                    f.write(" ".join(step_data) + "\n")

                self.write_queue.task_done()

    def encode_file(self, input_file_path, output_compressed_path):
        print(f"\n--- Medusa Rank Encoding (训练head总数={self.num_heads}, "
              f"本次使用head数={self.num_active_heads}, step_size={self.step_size}) ---")

        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)

        B = self.args.batch_size
        # chunk_length 需要能被 step_size 整除（简化边界处理）
        chunk_length = len(text_tokens) // B
        # 让 (chunk_length - 1) 是 step_size 的整数倍，便于整块处理
        usable_len = ((chunk_length - 1) // self.step_size) * self.step_size + 1
        chunk_length = usable_len
        valid_token_len = chunk_length * B
        text_tokens = text_tokens[:valid_token_len]

        batches = torch.tensor(text_tokens, device=self.device).view(B, chunk_length)
        first_tokens = batches[:, 0].cpu().numpy()

        num_real_steps = chunk_length - 1  # 总共要还原多少个后续 token

        writer_thread = threading.Thread(
            target=self._writer_worker,
            args=(output_compressed_path, B, num_real_steps, first_tokens)
        )
        writer_thread.start()

        input_ids        = batches[:, 0].unsqueeze(1)   # [B, 1]
        past_key_values  = None
        MAX_L = self.args.context_length
        W     = self.args.window_length

        # ---- 预热一次前向，排除首轮 cuDNN autotune / allocator 一次性开销，然后清零峰值显存 ----
        warmup_and_reset_peak_memory(
            self.model, self.medusa_heads, input_ids, self.device, self.num_active_heads
        )

        step = 0  # token位置指针（已经生成到哪个位置）
        t0 = time.perf_counter()

        with torch.inference_mode():
            while step < chunk_length - 1:
                # 本次最多能推进 step_size 个 token，但不能超过剩余长度
                cur_step_size = min(self.step_size, chunk_length - 1 - step)

                position_ids = torch.full(
                    (B, 1), step, dtype=torch.long, device=self.device
                )

                if past_key_values is not None and get_kv_len(past_key_values) >= MAX_L:
                    past_key_values = slice_kv_cache(past_key_values, W)

                # ---- 一次 forward，拿 hidden_states + base logits ----
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    output_hidden_states=True
                )
                hidden_states    = outputs.hidden_states[-1][:, -1, :]  # [B, hidden]
                past_key_values  = outputs.past_key_values

                # ---- Medusa heads 一次性出 num_active_heads 个位置的 logits ----
                # 只跑前 num_active_heads 个 head，多余的 head 不参与本次压缩
                medusa_logits_list = self.medusa_heads(
                    hidden_states.unsqueeze(1), num_heads=self.num_active_heads
                )

                all_logits = []
                for k in range(self.num_active_heads):
                    all_logits.append(medusa_logits_list[k][:, -1, :])
                # pos_offset=0 → head0 预测 step+1
                # pos_offset=1 → head1 预测 step+2

                # ---- 逐位置计算 rank，并收集本批要写入的数据 ----
                batch_write_items = []
                actual_tokens_this_round = []  # 真实token，用于后面更新input_ids

                for pos_offset in range(cur_step_size):
                    target_tokens = batches[:, step + 1 + pos_offset]  # [B]
                    logits_this   = all_logits[pos_offset]              # [B, V]

                    topk_values, sorted_indices = torch.topk(logits_this, k=32, dim=-1)
                    matches    = (sorted_indices == target_tokens.unsqueeze(1))
                    ranks      = matches.int().argmax(dim=-1) + 1
                    not_found  = ~matches.any(dim=-1)
                    ranks[not_found] = 33

                    batch_write_items.append(
                        (ranks.cpu().numpy(), target_tokens.cpu().numpy())
                    )
                    actual_tokens_this_round.append(target_tokens)

                self.write_queue.put(batch_write_items)

                # ---- 关键：用真实token（teacher forcing）推进下一轮的input_ids ----
                # 因为我们有完整原文，下一轮的输入就是这批的最后一个真实token
                # KV Cache 需要补齐中间这 cur_step_size-1 个 token 的信息！
                # 由于只 forward 了 1 次（只算了最后一个位置的 hidden），
                # KV Cache 里还缺中间 token 的 K/V，必须用 prefill 补齐
                if cur_step_size > 1:
                    # 把 actual_tokens_this_round[0:-1] 这些"中间"token 一次性 prefill
                    # 补齐 KV Cache（因为这些 token 的 K/V 还没有写入 past_key_values）
                    mid_tokens = torch.stack(
                        actual_tokens_this_round[:-1], dim=1
                    )  # [B, cur_step_size-1]
                    mid_pos = torch.arange(
                        step + 1, step + cur_step_size,
                        dtype=torch.long, device=self.device
                    ).unsqueeze(0).expand(B, -1)

                    mid_out = self.model(
                        input_ids=mid_tokens,
                        past_key_values=past_key_values,
                        position_ids=mid_pos,
                        use_cache=True
                    )
                    past_key_values = mid_out.past_key_values

                # 最后一个真实token作为下一轮的输入
                input_ids = actual_tokens_this_round[-1].unsqueeze(1)

                step += cur_step_size

                if step % 100 < self.step_size:
                    elapsed = time.perf_counter() - t0
                    speed   = step * B / elapsed if elapsed > 0 else 0
                    print(f"\rEncoding {step}/{chunk_length-1} | "
                          f"speed={speed:.0f} tok/s | Queue={self.write_queue.qsize()}",
                          end="")

        self.write_queue.put(None)
        writer_thread.join()
        total_time = time.perf_counter() - t0

        # ---- 记录 encode 阶段的 peak reserved 显存 ----
        self.peak_vram_gb = get_peak_reserved_gb(self.device)

        self._print_stats(text, text_tokens, total_time)
        return text_tokens

    def _print_stats(self, original_text, processed_tokens, total_time):
        actual_text = self.tokenizer.decode(processed_tokens, skip_special_tokens=True)
        original_bytes = len(actual_text.encode('utf-8'))
        original_bits = original_bytes * 8

        c = self.final_counts
        total_tokens = sum(c.values())
        expected_bits = (c[1] * 2 + c[2] * 4 + c[3] * 7 + c[4] * 20)
        bits_per_byte = expected_bits / original_bytes if original_bytes > 0 else float('nan')

        encode_speed_kbps = (original_bytes / 1024) / total_time if total_time > 0 else float('nan')

        print(f"\n\n{'='*45}")
        print(f"Medusa Rank Compression 统计 (耗时: {total_time:.2f}s)")
        print(f"{'='*45}")
        print(f"使用 head 数: {self.num_active_heads} / {self.num_heads} (训练总数)")
        print(f"吞吐量: {len(processed_tokens) / total_time:.1f} tokens/s")
        print(f"Encode 速度: {encode_speed_kbps:.2f} KB/s")
        print(f"forward次数节省: 约 {self.step_size}x (每次forward处理{self.step_size}个位置)")
        print(f"{'-'*45}")
        print(f"命中率（按位置聚合，含head 0-{self.num_active_heads}）:")
        for i in range(1, 5):
            print(f"  Index {i}: {c[i]/total_tokens:>6.2%} ({c[i]} tokens)")
        print(f"{'-'*45}")
        print(f"原始文本: {original_bits:>12,} bits ({original_bytes:,} bytes)")
        print(f"预估压缩: {expected_bits:>12,} bits")
        print(f"压缩率(bit/byte): {bits_per_byte:.4f} bit/byte")
        print(f"理论压缩率: {original_bits/expected_bits:>10.4f}x")
        if self.peak_vram_gb is not None:
            print(f"Encode 阶段 Peak GPU 显存 (max_memory_reserved): {self.peak_vram_gb:.3f} GB")
        else:
            print(f"Encode 阶段 Peak GPU 显存: N/A (当前设备非 CUDA)")
        print(f"{'='*45}")


# ==========================================
# Decoder
# ==========================================

class MedusaRankDecoder:
    """
    与 Encoder 对称：一次 forward 拿到 base + N 个位置的 logits，
    用压缩文件里存的 rank 信息，逐位置还原 token。

    注意：这里的 N（即 step_size / 本次实际用到的 head 数）直接从压缩文件的 header 里读取，
    不需要额外传参 —— 无论当初压缩时用了几个 head，都能自动对齐，
    不会出现"训练4个head、压缩用了3个、解码却按4个跑"这种不匹配问题。

    decode 端用的是"预测的" token 来推进（不是真实token，因为我们还不知道），
    但 rank 信息保证了能从预测的 topk 里精确取出正确的 token——
    这正是 rank 编码无损的核心，不依赖 Medusa head 预测准不准。
    """

    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model        = model
        self.medusa_heads = medusa_heads
        self.tokenizer    = tokenizer
        self.device       = device
        self.args         = args
        self.peak_vram_gb = None   # decode 阶段 peak reserved 显存（GB）

    def decode_file(self, input_compressed_path, output_text_path):
        print(f"\n--- Medusa Rank Decoding: {input_compressed_path} ---")

        with open(input_compressed_path, 'r') as f:
            header = f.readline().split()
            batch_size, num_real_steps, window_length, step_size = map(int, header)
            first_tokens = list(map(int, f.readline().split()))

            if step_size > self.medusa_heads.num_heads:
                raise ValueError(
                    f"压缩文件要求 step_size={step_size} 个 head，"
                    f"但当前加载的 medusa_heads 只有 {self.medusa_heads.num_heads} 个，"
                    f"请确认加载的是训练时同一套（或更多head数）的 medusa_heads_dir。"
                )

            chunk_length = num_real_steps + 1
            decoded_ids = torch.zeros(
                (batch_size, chunk_length), dtype=torch.long, device=self.device
            )
            decoded_ids[:, 0] = torch.tensor(first_tokens, device=self.device)

            input_ids        = decoded_ids[:, 0].unsqueeze(1)
            past_key_values  = None
            MAX_L = self.args.context_length
            W     = window_length

            # 预读所有行（rank信息），按 step_size 一组分组
            all_lines = [line.split() for line in f]

        print(f"压缩文件记录的 step_size(压缩时使用的head数) = {step_size}")

        # ---- 预热一次前向，排除首轮 cuDNN autotune / allocator 一次性开销，然后清零峰值显存 ----
        warmup_and_reset_peak_memory(
            self.model, self.medusa_heads, input_ids, self.device, step_size
        )

        step = 0
        line_idx = 0  # 指向 all_lines 的位置
        t0 = time.perf_counter()

        with torch.inference_mode():
            while step < chunk_length - 1:
                cur_step_size = min(step_size, chunk_length - 1 - step)

                position_ids = torch.full(
                    (batch_size, 1), step, dtype=torch.long, device=self.device
                )

                if past_key_values is not None and get_kv_len(past_key_values) >= MAX_L:
                    past_key_values = slice_kv_cache(past_key_values, W)

                # ---- 一次 forward 拿 hidden_states + base logits ----
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                    output_hidden_states=True
                )
                hidden_states   = outputs.hidden_states[-1][:, -1, :]
                past_key_values = outputs.past_key_values

                # 只跑压缩时实际用到的 head 数（step_size），避免多余计算
                medusa_logits_list = self.medusa_heads(
                    hidden_states.unsqueeze(1), num_heads=step_size
                )

                all_logits = []
                for k in range(step_size):
                    all_logits.append(medusa_logits_list[k][:, -1, :])
                # pos_offset=0 → head0 预测 step+1
                # pos_offset=1 → head1 预测 step+2

                # ---- 逐位置还原 token ----
                decoded_tokens_this_round = []

                for pos_offset in range(cur_step_size):
                    logits_this = all_logits[pos_offset]
                    topk_indices = torch.topk(logits_this, k=32, dim=-1).indices  # [B, 32]

                    line = all_lines[line_idx]
                    line_idx += 1

                    next_tokens = torch.zeros(
                        batch_size, dtype=torch.long, device=self.device
                    )
                    for b in range(batch_size):
                        parts = line[b].split(',')
                        idx = int(parts[0])
                        if idx == 1:
                            next_tokens[b] = topk_indices[b, 0]
                        elif idx in [2, 3]:
                            rank = int(parts[1])
                            next_tokens[b] = topk_indices[b, rank - 1]
                        else:
                            next_tokens[b] = int(parts[1])

                    decoded_ids[:, step + 1 + pos_offset] = next_tokens
                    decoded_tokens_this_round.append(next_tokens)

                # ---- 用刚还原出的token补齐KV Cache（除最后一个外）----
                if cur_step_size > 1:
                    mid_tokens = torch.stack(
                        decoded_tokens_this_round[:-1], dim=1
                    )  # [B, cur_step_size-1]
                    mid_pos = torch.arange(
                        step + 1, step + cur_step_size,
                        dtype=torch.long, device=self.device
                    ).unsqueeze(0).expand(batch_size, -1)

                    mid_out = self.model(
                        input_ids=mid_tokens,
                        past_key_values=past_key_values,
                        position_ids=mid_pos,
                        use_cache=True
                    )
                    past_key_values = mid_out.past_key_values

                input_ids = decoded_tokens_this_round[-1].unsqueeze(1)
                step += cur_step_size

                if step % 100 < step_size:
                    print(f"\rDecoding {step}/{chunk_length-1}", end="")

        total_time = time.perf_counter() - t0

        # ---- 记录 decode 阶段的 peak reserved 显存 ----
        self.peak_vram_gb = get_peak_reserved_gb(self.device)

        final_token_list = decoded_ids.view(-1).tolist()
        decoded_text = self.tokenizer.decode(final_token_list, skip_special_tokens=True)
        decoded_bytes = len(decoded_text.encode('utf-8'))
        decode_speed_kbps = (decoded_bytes / 1024) / total_time if total_time > 0 else float('nan')

        print(f"\n解码耗时: {total_time:.2f}s | "
              f"速度: {(batch_size*chunk_length)/total_time:.1f} tokens/s | "
              f"{decode_speed_kbps:.2f} KB/s")
        if self.peak_vram_gb is not None:
            print(f"Decode 阶段 Peak GPU 显存 (max_memory_reserved): {self.peak_vram_gb:.3f} GB")
        else:
            print(f"Decode 阶段 Peak GPU 显存: N/A (当前设备非 CUDA)")

        with open(output_text_path, 'w', encoding='utf-8') as f_out:
            f_out.write(decoded_text)

        return final_token_list


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Medusa-accelerated rank compression and decompression script.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or local path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to fine-tuned LoRA weights (optional).")
    parser.add_argument("--medusa_heads_dir", type=str, required=True,
                        help="Directory path containing trained Medusa heads.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"],
                        help="Model precision: fp16 / fp32 / bf16")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing.")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length.")
    parser.add_argument("--window_length", type=int, default=1024, help="Sliding window length.")
    parser.add_argument("--num_active_heads", type=int, default=None,
                        help="Number of active Medusa heads (first n) to use during compression/decompression. "
                             "Defaults to all heads saved in the trained directory. "
                             "Must satisfy 1 <= n <= total_trained_heads.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw text input file.")
    parser.add_argument("--compressed_file", type=str, default="medusa_compressed_data.txt", help="Path to output compressed binary file.")
    parser.add_argument("--output_file", type=str, default="medusa_decoded_text.txt", help="Path to output reconstructed text file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[args.dtype]

    if device == "cpu" and args.dtype == "fp16":
        print("Warning: fp16 has poor performance/support on CPU. Consider using fp32 or bf16 instead.")

    print(f"Loading Model: {args.model_name} on {device} (dtype={args.dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=device
    )
    if args.lora_path:
        print(f"Applying LoRA weights from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=dtype)
        model = model.merge_and_unload()
    model.eval()

    print(f"Loading Medusa heads from {args.medusa_heads_dir} (dtype={args.dtype})")
    medusa_heads = MedusaHeadCollection.load(args.medusa_heads_dir, device, dtype=dtype)
    medusa_heads.eval()

    # ---- Validation for num_active_heads ----
    if args.num_active_heads is None:
        args.num_active_heads = medusa_heads.num_heads
    if not (1 <= args.num_active_heads <= medusa_heads.num_heads):
        raise ValueError(
            f"Invalid --num_active_heads={args.num_active_heads}. "
            f"Must be within range [1, {medusa_heads.num_heads}] "
            f"({medusa_heads.num_heads} is total trained heads in {args.medusa_heads_dir})."
        )
    print(f"Active heads configured: {args.num_active_heads} / {medusa_heads.num_heads}")

    encoder = MedusaRankEncoder(model, medusa_heads, tokenizer, device, args)
    decoder = MedusaRankDecoder(model, medusa_heads, tokenizer, device, args)

    original_processed_tokens = encoder.encode_file(args.input_file, args.compressed_file)
    decoded_token_ids = decoder.decode_file(args.compressed_file, args.output_file)

    is_lossless = (original_processed_tokens == decoded_token_ids)
    print(f"\n{'*'*30}")
    print(f"Lossless Verification: {'PASSED' if is_lossless else 'FAILED (Mismatch)'}")
    if not is_lossless:
        print(f"Length Mismatch: Original ({len(original_processed_tokens)}) vs Decoded ({len(decoded_token_ids)})")
        for i in range(min(len(original_processed_tokens), len(decoded_token_ids))):
            if original_processed_tokens[i] != decoded_token_ids[i]:
                print(f"First mismatch at index: {i}")
                print(f"Original: {original_processed_tokens[i]} | Decoded: {decoded_token_ids[i]}")
                break
    print('*' * 30)


if __name__ == "__main__":
    main()
