import os
import time
import queue
import argparse
import threading
import contextlib
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# 复用已经验证过的 batch-invariant kernel。这次不需要像 Medusa 那样做
# "轮次对齐"的结构性改造——因为 batch invariant 的本意就是"不管这次调用
# 一起处理多少个位置，同一个位置的结果都一样"，所以 encode（固定 gamma
# 分块）和 decode（数据依赖的 S/L 连续段）这两种完全不对称的切分方式，
# 只要 draft/target 两个模型都套上这套 kernel，就不需要互相对齐 shape。
try:
    from batch_invariant_attention import (
        fp32_batch_invariant_attention_forward,
        fixed_tile_attention_forward,
        batch_invariant_linear_mode,
    )
    _HAS_BATCH_INVARIANT = True
    _BATCH_INVARIANT_IMPORT_ERROR = None
except ImportError as e:
    _HAS_BATCH_INVARIANT = False
    _BATCH_INVARIANT_IMPORT_ERROR = e


DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def apply_precision_guards():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def register_attention_patch(attn_fn, label):
    """跟 fast_rank_compresssion.py / medusa 版本里的同名函数逻辑一致。
    因为是替换 Qwen2Attention 这个类本身的方法（不是给某个模型实例打补丁），
    draft_model 和 target_model 只要都是 Qwen2 架构，一次注册两个模型都生效。"""
    patched_registry = False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["eager"] = attn_fn
        patched_registry = True
    except Exception as e:
        print(f"[{label}] 通过 ALL_ATTENTION_FUNCTIONS 打补丁失败（可能是 transformers 版本较旧）：{e}")

    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_module
        if hasattr(qwen2_module, "eager_attention_forward"):
            qwen2_module.eager_attention_forward = attn_fn
            patched_registry = True
    except Exception as e:
        print(f"[{label}] 通过替换 modeling_qwen2.eager_attention_forward 打补丁失败：{e}")

    patched_class = False
    try:
        import transformers.models.qwen2.modeling_qwen2 as qwen2_module
        if not hasattr(qwen2_module, "eager_attention_forward"):
            def _patched_qwen2_attention_forward(
                self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None,
            ):
                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                query_states, key_states = qwen2_module.apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )

                scale = 1.0 / (self.head_dim ** 0.5)
                attn_output, attn_weights = attn_fn(
                    self, query_states, key_states, value_states, attention_mask, scale,
                    dropout=(self.attention_dropout if self.training else 0.0),
                )
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
                attn_output = self.o_proj(attn_output)

                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights, past_key_value

            qwen2_module.Qwen2Attention.forward = _patched_qwen2_attention_forward
            patched_class = True
    except Exception as e:
        print(f"[{label}] 通过整体替换 Qwen2Attention.forward 打补丁失败：{e}")

    if not (patched_registry or patched_class):
        raise RuntimeError(f"{label} 补丁未能生效，所有打补丁方式都失败了，请检查 transformers 版本")
    print(f"[{label}] 已启用（同时覆盖 draft/target 两个模型）" +
          ("（整体替换 Qwen2Attention.forward）" if patched_class else "（通过注册表/模块函数）"))


def slice_kv_cache(past_key_values, window_length):
    """裁剪 KV Cache，同时更新 DynamicCache 内部计数"""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        past_key_values.crop(window_length)
        return past_key_values
    else:
        return tuple((k[..., -window_length:, :], v[..., -window_length:, :]) for k, v in past_key_values)


def get_kv_cache_length(past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].size(2)


def get_ranks_and_bins_fast(logits, target_tokens):
    """
    极致性能版：全向量化计算，支持多维张量 [batch, cur_gamma, vocab]
    彻底消除 GPU 到 CPU 的同步等待开销 (.item() 和 for 循环)
    """
    if logits.dtype != torch.float32:
        logits = logits.float()

    topk_vals, topk_indices = torch.topk(logits, k=32, dim=-1)
    expanded_targets = target_tokens.unsqueeze(-1)
    matches = (topk_indices == expanded_targets)

    found = matches.any(dim=-1)
    rank_indices = matches.int().argmax(dim=-1)
    ranks = torch.where(found, rank_indices + 1, torch.tensor(33, device=logits.device))

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
        self.write_queue = queue.Queue(maxsize=500)
        self.final_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.model_usage = {'S': 0, 'L': 0}

    def _writer_worker(self, output_path, batch_size, chunk_length, first_tokens):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{batch_size} {chunk_length} {self.args.window_length}\n")
            f.write(" ".join(map(str, first_tokens)) + "\n")

            while True:
                item = self.write_queue.get()
                if item is None:
                    break

                d_ranks_cpu, d_bins_cpu, t_ranks_cpu, t_bins_cpu, targets_cpu = item
                step_data = []

                for b in range(batch_size):
                    if d_ranks_cpu[b] == 1:
                        self.model_usage['S'] += 1
                        step_data.append("1")
                    else:
                        self.model_usage['L'] += 1
                        idx = t_bins_cpu[b]
                        r = t_ranks_cpu[b]
                        self.final_counts[idx] += 1

                        if idx == 1:
                            val = 0
                        elif idx == 4:
                            val = int(targets_cpu[b])
                        else:
                            val = int(r)

                        step_data.append(f"0,{idx},{val}")

                f.write(" ".join(step_data) + "\n")
                self.write_queue.task_done()

    def encode_file(self, input_file_path, output_compressed_path):
        print(f"\n--- Starting Speculative Encoding (batch-invariant kernel, {self.args.dtype}) ---")
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

                # ---- 块级别上下文对齐 ----
                if draft_past is not None:
                    curr_len = get_kv_cache_length(draft_past)
                    if curr_len >= MAX_L:
                        draft_past = slice_kv_cache(draft_past, W)
                        target_past = slice_kv_cache(target_past, W)

                input_seq = batches[:, step: step + cur_gamma]
                pos_ids_block = torch.arange(
                    step, step + cur_gamma, dtype=torch.long, device=self.device
                ).unsqueeze(0).expand(self.args.batch_size, -1)

                # ---- Target model：一次并行 teacher-forcing 处理整块 ----
                t_out = self.target_model(
                    input_ids=input_seq,
                    past_key_values=target_past,
                    position_ids=pos_ids_block,
                    use_cache=True
                )
                target_logits = t_out.logits
                target_past = t_out.past_key_values

                # ---- Draft model：改成跟 target 一样的一次并行调用 ----
                # 原来是 for 循环逐 token 算，现在有了真正 batch-invariant
                # 的 kernel 兜底，不需要跟 decode 端的 shape 对齐，直接
                # 照抄 target 的写法即可，白得一次加速。
                d_out = self.draft_model(
                    input_ids=input_seq,
                    past_key_values=draft_past,
                    position_ids=pos_ids_block,
                    use_cache=True
                )
                draft_logits = d_out.logits
                draft_past = d_out.past_key_values

                # ---- 向量化计算 Rank 并分发 ----
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
        total_tokens = sum(m.values())

        expected_bits = (
                m['S'] * 1 +
                m['L'] * (1 + 2) +
                c[2] * 2 +
                c[3] * 5 +
                c[4] * 18
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
# 解码器类 (Speculative Decoder BS=1)
#
# 不需要改动：draft 在 S run 里自己生成自己（天然顺序依赖，无法并行），
# target 在 L run 里自己生成自己，也是同理——这两段是真正意义上"不可能
# 消除"的顺序依赖，跟 Rank Coding 的 decode 是同一类限制。sync 环节
# 已经是单次合并调用，没有能再合并的空间（不像 Medusa 那样有两次
# 冗余调用可以合并）。
# ==========================================
class FastRankDecoder:
    def __init__(self, draft_model, target_model, tokenizer, device, args):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

    def _parse_all_instructions(self, instructions):
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
            else:
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

        instructions = [line.strip().split()[target_batch_idx] for line in lines[2:] if line.strip()]
        runs = self._parse_all_instructions(instructions)

        decoded_ids = torch.zeros((1, chunk_length), dtype=torch.long, device=self.device)
        decoded_ids[0, 0] = first_token

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

                if draft_past is not None and get_kv_cache_length(draft_past) >= MAX_L:
                    draft_past = slice_kv_cache(draft_past, W)
                if target_past is not None and get_kv_cache_length(target_past) >= MAX_L:
                    target_past = slice_kv_cache(target_past, W)

                if run_type == "S":
                    curr_input = decoded_ids[:, step: step + 1]

                    for k in range(K):
                        curr_pos = global_positions[:, step + k: step + k + 1]
                        d_out = self.draft_model(
                            input_ids=curr_input, past_key_values=draft_past,
                            position_ids=curr_pos, use_cache=True
                        )
                        draft_past = d_out.past_key_values
                        next_token = torch.argmax(d_out.logits[0, -1, :])
                        decoded_ids[0, step + k + 1] = next_token
                        curr_input = next_token.view(1, 1)

                    sync_input = decoded_ids[:, step: step + K]
                    sync_pos = global_positions[:, step: step + K]
                    t_out = self.target_model(
                        input_ids=sync_input, past_key_values=target_past,
                        position_ids=sync_pos, use_cache=True
                    )
                    target_past = t_out.past_key_values

                else:  # run_type == "L"
                    curr_input = decoded_ids[:, step: step + 1]

                    for k in range(K):
                        curr_pos = global_positions[:, step + k: step + k + 1]
                        t_out = self.target_model(
                            input_ids=curr_input, past_key_values=target_past,
                            position_ids=curr_pos, use_cache=True
                        )
                        target_past = t_out.past_key_values

                        logits = t_out.logits[0, -1, :]
                        idx, val = run_data[k]

                        if idx == 1:
                            next_token = torch.argmax(logits)
                        elif idx in [2, 3]:
                            topk_indices = torch.topk(logits, k=32).indices
                            next_token = topk_indices[val - 1]
                        else:
                            next_token = torch.tensor(val, device=self.device)

                        decoded_ids[0, step + k + 1] = next_token
                        curr_input = next_token.view(1, 1)

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
    parser.add_argument("--gamma", type=int, default=64, help="投机步长 N")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--window_length", type=int, default=512)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--compressed_file", type=str, default="output.npz")
    parser.add_argument("--output_file", type=str, default="decoded_output.txt")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--attn_implementation", type=str, default="eager",
                         choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--batch_invariant_attention", action="store_true")
    parser.add_argument("--fixed_tile_attention", action="store_true")
    parser.add_argument("--batch_invariant_linear", action="store_true")
    args = parser.parse_args()

    if (args.batch_invariant_attention or args.fixed_tile_attention or args.batch_invariant_linear) \
            and not _HAS_BATCH_INVARIANT:
        raise ImportError(
            f"未能导入 batch_invariant_attention.py：{_BATCH_INVARIANT_IMPORT_ERROR!r}\n"
            "确认该文件跟本脚本放在同一目录下。"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[args.dtype]

    apply_precision_guards()
    # 只需要注册一次：Qwen2Attention.forward 是类级别的替换，
    # draft_model 和 target_model 只要都是 Qwen2 架构，会同时生效。
    if args.fixed_tile_attention:
        register_attention_patch(fixed_tile_attention_forward, "fixed_tile_attention")
    elif args.batch_invariant_attention:
        register_attention_patch(fp32_batch_invariant_attention_forward, "batch_invariant_attention")

    print(f"Loading Draft Model: {args.draft_model_name} ...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_name, torch_dtype=dtype, device_map=device,
        attn_implementation=args.attn_implementation,
    ).eval()

    lora_model_path = ("/root/yanzhe/yanzhe/FineZip-main/finezip/finetuned_models/"
                       "h100/Qwen0.5b/wiki_05_newloss_128_r16")
    # Lora权重,不用的时候记得注释掉下两行
    # draft_model = PeftModel.from_pretrained(draft_model, lora_model_path, torch_dtype=dtype)
    # draft_model = draft_model.merge_and_unload()

    print(f"Loading Target Model: {args.target_model_name} ...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name, torch_dtype=dtype, device_map=device,
        attn_implementation=args.attn_implementation,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name)

    encoder = FastRankEncoder(draft_model, target_model, tokenizer, device, args)
    decoder = FastRankDecoder(draft_model, target_model, tokenizer, device, args)

    linear_ctx = batch_invariant_linear_mode() if args.batch_invariant_linear else contextlib.nullcontext()
    with linear_ctx:
        all_original_tokens = list(encoder.encode_file(args.input_file, args.compressed_file))
        chunk_length = len(all_original_tokens) // args.batch_size
        original_batch_0 = all_original_tokens[:chunk_length]
        decoded_batch_0 = list(decoder.decode_file(args.compressed_file, target_batch_idx=0))

    decoded_text = tokenizer.decode(decoded_batch_0, skip_special_tokens=True)
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(decoded_text)
    print(f"\n解码文本已写入: {args.output_file}")

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
