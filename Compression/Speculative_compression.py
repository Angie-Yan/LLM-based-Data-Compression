import os
import time
import queue
import argparse
import threading
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache



def slice_kv_cache(past_key_values, window_length):
    """裁剪 KV Cache，同时更新 DynamicCache 内部计数"""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        # 使用内置 crop 方法，保证 _seen_tokens 同步
        past_key_values.crop(window_length)
        return past_key_values
    else:
        return tuple((k[..., -window_length:, :], v[..., -window_length:, :]) for k, v in past_key_values)


def get_ranks_and_bins_fast(logits, target_tokens):
    """
    极致性能版：全向量化计算，支持多维张量 [batch, cur_gamma, vocab]
    彻底消除 GPU 到 CPU 的同步等待开销 (.item() 和 for 循环)
    """
    if logits.dtype != torch.float32:
        logits = logits.float()

    # topk_indices: [..., 32]
    topk_vals, topk_indices = torch.topk(logits, k=32, dim=-1)

    # 将 targets 扩展以匹配 topk 的最后一个维度: [..., 1]
    expanded_targets = target_tokens.unsqueeze(-1)
    matches = (topk_indices == expanded_targets)

    # 找到是否存在匹配，以及匹配的索引
    found = matches.any(dim=-1)
    rank_indices = matches.int().argmax(dim=-1)

    # 计算 Rank (1-based)，找不到的给 33
    ranks = torch.where(found, rank_indices + 1, torch.tensor(33, device=logits.device))

    # 向量化计算 bins
    bins = torch.full_like(target_tokens, 4)
    bins = torch.where(found & (ranks == 1), 1, bins)
    bins = torch.where(found & (ranks >= 2) & (ranks <= 4), 2, bins)
    bins = torch.where(found & (ranks >= 5) & (ranks <= 32), 3, bins)

    return ranks, bins


# ==========================================
# 编码器类 (Speculative Encoder)
# ==========================================
class FastRankEncoder:
    def __init__(self, draft_model, target_model, tokenizer, device, args):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.write_queue = queue.Queue(maxsize=500)  # 调大队列，防止向量化提速后阻塞
        self.final_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.model_usage = {'S': 0, 'L': 0}

    def _writer_worker(self, output_path, batch_size, chunk_length, first_tokens):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{batch_size} {chunk_length} {self.args.window_length}\n")
            f.write(" ".join(map(str, first_tokens)) + "\n")

            while True:
                item = self.write_queue.get()
                if item is None: break

                d_ranks_cpu, d_bins_cpu, t_ranks_cpu, t_bins_cpu, targets_cpu = item
                step_data = []

                for b in range(batch_size):
                    # 1. 新判定逻辑：小模型 Rank=1 则命中，否则由大模型接管
                    if d_ranks_cpu[b] == 1:
                        self.model_usage['S'] += 1
                        # 小模型命中，仅存 1-bit 标志位 "1"
                        step_data.append("1")
                    else:
                        self.model_usage['L'] += 1
                        idx = t_bins_cpu[b]
                        r = t_ranks_cpu[b]

                        # 仅统计大模型的 bin 分布，用于计算实际开销
                        self.final_counts[idx] += 1

                        # 大模型根据 bin 存具体的额外信息
                        if idx == 1:
                            val = 0
                        elif idx == 4:
                            val = int(targets_cpu[b])
                        else:
                            val = int(r)

                        # 大模型存储：前缀 0 + bin + rank/ID
                        step_data.append(f"0,{idx},{val}")

                f.write(" ".join(step_data) + "\n")
                self.write_queue.task_done()

    def encode_file(self, input_file_path, output_compressed_path):
        print(f"\n--- Starting Speculative Encoding (Context Synchronized) ---")
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)
        Max_token = 100000
        text_tokens = text_tokens[: Max_token]
        chunk_length = len(text_tokens) // self.args.batch_size
        text_tokens = text_tokens[:chunk_length * self.args.batch_size]

        batches = torch.tensor(text_tokens, device=self.device).view(self.args.batch_size, chunk_length)
        first_tokens = batches[:, 0].cpu().numpy()

        writer_thread = threading.Thread(target=self._writer_worker, args=(
            output_compressed_path, self.args.batch_size, chunk_length, first_tokens))
        writer_thread.start()

        draft_past = None
        target_past = None
        step = 0
        MAX_L = self.args.context_length
        W = self.args.window_length

        t0 = time.perf_counter()
        with torch.inference_mode():
            while step < chunk_length - 1:
                cur_gamma = min(self.args.gamma, chunk_length - 1 - step)

                # ==========================================
                # 1. 块级别上下文对齐
                # ==========================================
                if draft_past is not None:
                    curr_len = draft_past.get_seq_length() if hasattr(draft_past, "get_seq_length") else draft_past[0][
                        0].size(2)
                    if curr_len >= MAX_L:
                        draft_past = slice_kv_cache(draft_past, W)
                        target_past = slice_kv_cache(target_past, W)

                # ==========================================
                # 2. 大模型 Block Forward
                # ==========================================
                input_seq = batches[:, step: step + cur_gamma]
                pos_ids_block = torch.arange(
                    step, step + cur_gamma, dtype=torch.long, device=self.device
                ).unsqueeze(0).expand(self.args.batch_size, -1)

                t_out = self.target_model(
                    input_ids=input_seq,
                    past_key_values=target_past,
                    position_ids=pos_ids_block,
                    use_cache=True
                )
                target_logits = t_out.logits
                target_past = t_out.past_key_values

                # 逐 token forward模式
                # target_logits_list = []
                # for i in range(cur_gamma):
                #     curr_input = batches[:, step + i].unsqueeze(1)
                #     current_pos_id = torch.tensor([[step + i]] * self.args.batch_size, device=self.device)
                #
                #     t_out = self.target_model(
                #         input_ids=curr_input,
                #         past_key_values=target_past,
                #         position_ids=current_pos_id,
                #         use_cache=True
                #     )
                #     target_logits_list.append(t_out.logits)
                #     target_past = t_out.past_key_values
                # target_logits = torch.cat(target_logits_list, dim=1)

                # ==========================================
                # 3. 小模型单步 Forward 循环
                # ==========================================
                draft_logits_list = []
                for i in range(cur_gamma):
                    curr_input = batches[:, step + i].unsqueeze(1)
                    current_pos_id = torch.tensor([[step + i]] * self.args.batch_size,
                                                  dtype=torch.long, device=self.device)

                    d_out = self.draft_model(
                        input_ids=curr_input,
                        past_key_values=draft_past,
                        position_ids=current_pos_id,
                        use_cache=True
                    )
                    draft_logits_list.append(d_out.logits)
                    draft_past = d_out.past_key_values

                draft_logits = torch.cat(draft_logits_list, dim=1)

                # ==========================================
                # 4. 向量化计算 Rank 并分发
                # ==========================================
                t_toks_seq = batches[:, step + 1: step + 1 + cur_gamma]
                d_ranks_seq, d_bins_seq = get_ranks_and_bins_fast(draft_logits, t_toks_seq)
                t_ranks_seq, t_bins_seq = get_ranks_and_bins_fast(target_logits, t_toks_seq)

                for i in range(cur_gamma):
                    self.write_queue.put((
                        d_ranks_seq[:, i].cpu().numpy(),
                        d_bins_seq[:, i].cpu().numpy(),
                        t_ranks_seq[:, i].cpu().numpy(),
                        t_bins_seq[:, i].cpu().numpy(),
                        t_toks_seq[:, i].cpu().numpy()
                    ))

                step += cur_gamma
                if step % 128 == 0 or step == chunk_length - 1:
                    print(f"\rEncoding {step}/{chunk_length - 1} | Queue: {self.write_queue.qsize()}", end="")

        self.write_queue.put(None)
        writer_thread.join()
        self._print_stats(text, text_tokens, time.perf_counter() - t0)
        return text_tokens

    def _print_stats(self, original_text, processed_tokens, total_time):
        actual_text = self.tokenizer.decode(processed_tokens, skip_special_tokens=True)
        original_bits = len(actual_text.encode('utf-8')) * 8
        c = self.final_counts
        m = self.model_usage
        total_tokens = sum(m.values())  # 总 Token 数量现在是 S 和 L 的总和

        # ==========================================
        # 严格计算新方案的预期 Bit 数
        # ==========================================
        # 1. 小模型命中区：每个 token 严格 1 bit
        # 2. 大模型未命中区：前缀位 (1 bit) + Bin 标签 (2 bits) = 3 bits
        #    加上各个 Bin 额外的 Rank / ID 存储开销
        expected_bits = (
                m['S'] * 1 +  # 小模型: "1"
                m['L'] * (1 + 2) +  # 大模型基底开销: 前缀 "0" + 2 bits Bin标签
                c[2] * 2 +  # 大模型 Bin 2: 额外 2 bits rank
                c[3] * 5 +  # 大模型 Bin 3: 额外 5 bits rank
                c[4] * 18  # 大模型 Bin 4: 额外 18 bits TokenID
        )

        print(f"\n\n" + "=" * 40)
        print(f"投机压缩统计报告 (Speculative Block Encode - 1 Bit Flag)")
        print(f"吞吐量: {len(processed_tokens) / total_time:.1f} tokens/s")
        print(f"模型分布: Small Hit={m['S']} ({m['S'] / total_tokens:.1%}), Large Miss={m['L']} ({m['L'] / total_tokens:.1%})")
        print(f"大模型分布：")
        print(f"rank=1: {c[1] / m['L']:.2%}")
        print(f"2<=rank<=4: {c[2] / m['L']:.2%}")
        print(f"5<=rank<=32: {c[3] / m['L']:.2%}")
        print(f"rank>32: {c[4] / m['L']:.2%}")

        print(f"-" * 40)
        print(f"   理论压缩率: {original_bits / expected_bits:>10.4f}x")
        print(f"   平均Bit/Token: {expected_bits / total_tokens:>10.4f} bits")
        print(f"=" * 40)



# ==========================================
# 解码器类 (Speculative Decoder BS=1) - Refactored
# ==========================================
class FastRankDecoder:
    def __init__(self, draft_model, target_model, tokenizer, device, args):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

    def _parse_all_instructions(self, instructions):
        """
        [优化 1] 全局预解析：一次性把所有指令转换为机器友好的格式，消除推理时的字符串处理。
        返回格式: list of (run_type, parsed_data_list)
        - S 模式: parsed_data_list 只有长度信息 (不需要解析指令)
        - L 模式: parsed_data_list 存储 [(idx, val), ...] 整数元组
        """
        if not instructions:
            return []

        runs = []
        curr_type = "S" if instructions[0] == "1" else "L"
        curr_run = []

        for inst in instructions:
            inst_type = "S" if inst == "1" else "L"

            if inst_type == "S":
                if curr_type == "S":
                    curr_run.append(None)
                else:
                    runs.append((curr_type, curr_run))
                    curr_type = "S"
                    curr_run = [None]
            else:  # inst_type == "L"
                parts = inst.split(',')
                parsed_inst = (int(parts[1]), int(parts[2]))
                if curr_type == "L":
                    curr_run.append(parsed_inst)
                else:
                    runs.append((curr_type, curr_run))
                    curr_type = "L"
                    curr_run = [parsed_inst]

        runs.append((curr_type, curr_run))
        return runs

    def decode_file(self, compressed_file_path, target_batch_idx=0):
        with open(compressed_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header = lines[0].strip().split()
        batch_size, chunk_length, window_length = map(int, header)
        first_token = int(lines[1].strip().split()[target_batch_idx])

        # 提取当前 Batch 的指令列表
        instructions = [line.strip().split()[target_batch_idx] for line in lines[2:] if line.strip()]

        # [优化 2] 取消 gamma 分组，直接获取全局的 runs
        # 这将极大增加连续 S 模式或 L 模式的 K 值，从而最大化平摊同步开销
        runs = self._parse_all_instructions(instructions)

        # [优化 3] 预分配连续显存，消灭 Python List 频繁 `.append()` 和 `torch.tensor()` 的开销
        decoded_ids = torch.zeros((1, chunk_length), dtype=torch.long, device=self.device)
        decoded_ids[0, 0] = first_token

        # 预生成全局的 position_ids，供后续通过切片零成本获取
        global_positions = torch.arange(chunk_length, device=self.device).unsqueeze(0)

        draft_past = None
        target_past = None
        MAX_L = self.args.context_length
        W = window_length
        step = 0

        start_time = time.perf_counter()

        with torch.inference_mode():
            for run_type, run_data in runs:
                K = len(run_data)

                # KV Cache 裁剪 (基于 O(1) 的 shape 检查，无同步开销)
                if draft_past is not None and draft_past[0][0].size(2) >= MAX_L:
                    draft_past = slice_kv_cache(draft_past, W)
                if target_past is not None and target_past[0][0].size(2) >= MAX_L:
                    target_past = slice_kv_cache(target_past, W)

                if run_type == "S":
                    # --- 小模型自回归生成 ---
                    # 直接从 pre-allocated tensor 中切片出输入，无需重新建 tensor
                    curr_input = decoded_ids[:, step: step + 1]

                    for k in range(K):
                        curr_pos = global_positions[:, step + k: step + k + 1]

                        d_out = self.draft_model(
                            input_ids=curr_input, past_key_values=draft_past,
                            position_ids=curr_pos, use_cache=True
                        )
                        draft_past = d_out.past_key_values

                        # 保持张量形态！不要使用 .item() 强行同步到 CPU
                        next_token = torch.argmax(d_out.logits[0, -1, :])

                        decoded_ids[0, step + k + 1] = next_token
                        curr_input = next_token.view(1, 1)  # 喂给下一步

                    # --- 大模型同步 (Prefill) ---
                    # 一次性将这段时间累积的 token 送入大模型同步 KV Cache
                    sync_input = decoded_ids[:, step: step + K]
                    sync_pos = global_positions[:, step: step + K]

                    t_out = self.target_model(
                        input_ids=sync_input, past_key_values=target_past,
                        position_ids=sync_pos, use_cache=True
                    )
                    target_past = t_out.past_key_values

                else:  # run_type == "L"
                    # --- 大模型自回归生成 ---
                    curr_input = decoded_ids[:, step: step + 1]

                    for k in range(K):
                        curr_pos = global_positions[:, step + k: step + k + 1]

                        t_out = self.target_model(
                            input_ids=curr_input, past_key_values=target_past,
                            position_ids=curr_pos, use_cache=True
                        )
                        target_past = t_out.past_key_values

                        logits = t_out.logits[0, -1, :]
                        idx, val = run_data[k]  # 预先解析好的指令，直接提取整数

                        if idx == 1:
                            next_token = torch.argmax(logits)
                        elif idx in [2, 3]:
                            # 仅在 GPU 上计算 TopK
                            topk_indices = torch.topk(logits, k=32).indices
                            next_token = topk_indices[val - 1]
                        else:  # idx == 4
                            # 将数字转为张量以保持在 GPU 上写入
                            next_token = torch.tensor(val, device=self.device)

                        decoded_ids[0, step + k + 1] = next_token
                        curr_input = next_token.view(1, 1)

                    # --- 小模型同步 (Prefill) ---
                    sync_input = decoded_ids[:, step: step + K]
                    sync_pos = global_positions[:, step: step + K]

                    d_out = self.draft_model(
                        input_ids=sync_input, past_key_values=draft_past,
                        position_ids=sync_pos, use_cache=True
                    )
                    draft_past = d_out.past_key_values

                step += K

        end_time = time.perf_counter()
        total_time = end_time - start_time

        final_token_list = decoded_ids[0].tolist()
        total_tokens = len(final_token_list)
        avg_speed = total_tokens / total_time if total_time > 0 else 0

        print(f"\n[Timing] Decode completed in {total_time:.4f} seconds")
        print(f"[Timing] Average speed: {avg_speed:.2f} tokens/second")

        return final_token_list


# ==========================================
# Main 执行流
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft_model_name", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen1.5-7B")
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--gamma", type=int, default=64, help="投机步长 N")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--window_length", type=int, default=512)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--compressed_file", type=str, default="output.npz")
    parser.add_argument("--output_file", type=str, default="decoded_output.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Draft Model: {args.draft_model_name} ...")
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_name, torch_dtype=torch.float32,
                                                       device_map=device).eval()

    lora_model_path = ("/root/yanzhe/yanzhe/FineZip-main/finezip/finetuned_models/"
                       "h100/Qwen0.5b/wiki_05_newloss_128_r16")
    # Lora权重,不用的时候记得注释掉
    draft_model = PeftModel.from_pretrained(draft_model, lora_model_path, torch_dtype=torch.float32)
    draft_model = draft_model.merge_and_unload()


    print(f"Loading Target Model: {args.target_model_name} ...")
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_name, torch_dtype=torch.float32,
                                                        device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)

    encoder = FastRankEncoder(draft_model, target_model, tokenizer, device, args)

    # Decoder 初始化时传入 args，以便获取 context_length
    decoder = FastRankDecoder(draft_model, target_model, tokenizer, device, args)

    # 1. 执行全量 Encode
    all_original_tokens = list(encoder.encode_file(args.input_file, args.compressed_file))

    # 由于原文被按 batch_size 切成了多个相互独立的段 (BS=16, ChunkLength=L)
    # 我们需要截取 Batch 0 的原始数据用来对比 (也就是前 chunk_length 个 token)
    chunk_length = len(all_original_tokens) // args.batch_size
    original_batch_0 = all_original_tokens[:chunk_length]

    # 2. 执行解码：指定只还原 Batch 0 (BS=1)
    decoded_batch_0 = list(decoder.decode_file(args.compressed_file, target_batch_idx=0))
    decoded_text = tokenizer.decode(decoded_batch_0, skip_special_tokens=True)
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(decoded_text)
    print(f"\n解码文本已写入: {args.output_file}")

    # 3. 对比无损性
    is_lossless = (original_batch_0 == decoded_batch_0)

    print(f"\n" + "*" * 40)
    print(f"无损验证 (Target Batch=0): {'通过 (Lossless)' if is_lossless else '失败 (Mismatch)'}")
    if not is_lossless:
        for i in range(min(len(original_batch_0), len(decoded_batch_0))):
            if original_batch_0[i] != decoded_batch_0[i]:
                print(f"首次不匹配位置: {i}")
                print(f"原值 TokenID: {original_batch_0[i]}")
                print(f"解码 TokenID: {decoded_batch_0[i]}")
                break
    print("*" * 40)

if __name__ == "__main__":
    main()
