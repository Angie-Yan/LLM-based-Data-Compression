import os
import time
import queue
import threading
import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


# ==========================================
# KV Cache clipping
# ==========================================
def slice_kv_cache(past_key_values, window_length):
    """裁剪 KV Cache 以实现 Sliding Window"""
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][..., -window_length:, :]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][..., -window_length:, :]
        return past_key_values
    else:
        sliced_tuples = tuple(
            (k[:, :, -window_length:, :], v[:, :, -window_length:, :])
            for k, v in past_key_values
        )
        return sliced_tuples


# ==========================================
# VRAM Measuring Function
# ==========================================
def warmup_and_reset_peak_memory(model, input_ids, device):
    if device != "cuda":
        return
    torch.cuda.synchronize()
    with torch.inference_mode():
        B = input_ids.size(0)
        dummy_pos = torch.zeros((B, 1), dtype=torch.long, device=device)
        _ = model(
            input_ids=input_ids,
            position_ids=dummy_pos,
            use_cache=True,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_reserved_gb(device):
    if device != "cuda":
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


# ==========================================
# Encoder
# ==========================================
class FastRankEncoder:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.write_queue = queue.Queue(maxsize=200)  # 缓冲区
        self.final_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    def _writer_worker(self, output_path, batch_size, chunk_length, first_tokens):
        """后台写入线程：负责字符串转换和磁盘写入"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入 Header
            f.write(f"{batch_size} {chunk_length} {self.args.window_length}\n")
            f.write(" ".join(map(str, first_tokens)) + "\n")

            while True:
                item = self.write_queue.get()
                if item is None: break

                ranks_cpu, targets_cpu = item
                step_data = []
                for b in range(batch_size):
                    r = ranks_cpu[b]
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
                        step_data.append(f"4,{targets_cpu[b]}")
                        self.final_counts[4] += 1

                f.write(" ".join(step_data) + "\n")
                self.write_queue.task_done()

    def encode_file(self, input_file_path, output_compressed_path):
        print(f"\n--- Starting asynchronous encoding: {input_file_path} ---")

        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text_tokens = self.tokenizer.encode(text, add_special_tokens=True)

        chunk_length = len(text_tokens) // self.args.batch_size
        valid_token_len = chunk_length * self.args.batch_size
        text_tokens = text_tokens[:valid_token_len]

        batches = torch.tensor(text_tokens, device=self.device).view(self.args.batch_size, chunk_length)
        first_tokens = batches[:, 0].cpu().numpy()

        # 启动后台写入线程
        writer_thread = threading.Thread(
            target=self._writer_worker,
            args=(output_compressed_path, self.args.batch_size, chunk_length, first_tokens)
        )
        writer_thread.start()

        input_ids = batches[:, 0].unsqueeze(1)
        past_key_values = None

        # --- 执行 warmup 并重置 peak VRAM ---
        warmup_and_reset_peak_memory(self.model, input_ids, self.device)

        # --- 初始化计时器 ---
        time_gpu_infer = 0.0
        time_gpu_rank = 0.0
        time_data_copy = 0.0
        time_cpu_queue = 0.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            for step in range(chunk_length - 1):
                position_ids = torch.full((self.args.batch_size, 1), step, dtype=torch.long, device=self.device)

                if past_key_values is not None:
                    # 兼容 DynamicCache 和 tuple
                    curr_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else \
                        past_key_values[0][0].size(2)
                    if curr_len >= self.args.context_length:
                        past_key_values = slice_kv_cache(past_key_values, self.args.window_length)

                # 1. 测 GPU Inference Time
                torch.cuda.synchronize()
                t_infer_start = time.perf_counter()

                outputs = self.model(input_ids=input_ids, past_key_values=past_key_values,
                                     position_ids=position_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                torch.cuda.synchronize()
                time_gpu_infer += (time.perf_counter() - t_infer_start)

                target_tokens = batches[:, step + 1]

                # 2. 测 GPU Rank 计算时间 (TopK)
                t_rank_start = time.perf_counter()
                topk_values, sorted_indices = torch.topk(logits, k=32, dim=-1)
                matches = (sorted_indices == target_tokens.unsqueeze(1))
                ranks = matches.int().argmax(dim=-1) + 1
                not_found = ~matches.any(dim=-1)
                ranks[not_found] = 33
                torch.cuda.synchronize()
                time_gpu_rank += (time.perf_counter() - t_rank_start)

                # 3. 测 Data Copy 时间 (Device to Host)
                t_copy_start = time.perf_counter()
                ranks_cpu = ranks.cpu().numpy()
                targets_cpu = target_tokens.cpu().numpy()
                time_data_copy += (time.perf_counter() - t_copy_start)

                # 4. 测 CPU 队列时间
                t_queue_start = time.perf_counter()
                self.write_queue.put((ranks_cpu, targets_cpu))
                time_cpu_queue += (time.perf_counter() - t_queue_start)

                input_ids = target_tokens.unsqueeze(1)

                if step % 100 == 0:
                    print(f"\rEncoding {step}/{chunk_length - 1} | Queue: {self.write_queue.qsize()}", end="")

        self.write_queue.put(None)
        writer_thread.join()
        total_time = time.perf_counter() - t0

        # 记录 Encode Peak VRAM
        peak_vram_gb = get_peak_reserved_gb(self.device)

        # --- Print Statistical Information ---
        print(f"\n\n[Encoder 耗时与显存分解]")
        print(f"  -> GPU 模型推理耗时: {time_gpu_infer:.3f} s ({time_gpu_infer / total_time:.1%})")
        print(f"  -> GPU Rank计算耗时: {time_gpu_rank:.3f} s ({time_gpu_rank / total_time:.1%})")
        print(f"  -> GPU->CPU 拷贝耗时: {time_data_copy:.3f} s ({time_data_copy / total_time:.1%})")
        print(f"  -> CPU 队列通信耗时: {time_cpu_queue:.3f} s ({time_cpu_queue / total_time:.1%})")
        print(f"  -> Encode Peak VRAM: {peak_vram_gb:.3f} GB")

        self._print_stats(text_tokens, total_time, peak_vram_gb)

        return text_tokens  # used to verify lossless

    def _print_stats(self, processed_tokens, total_time, peak_vram):
        actual_text = self.tokenizer.decode(processed_tokens, skip_special_tokens=True)
        original_bytes = len(actual_text.encode('utf-8'))
        original_bits = original_bytes * 8

        # calculate bits after compression
        c = self.final_counts
        total_tokens = sum(c.values())
        expected_bits = (c[1] * 2 + c[2] * 4 + c[3] * 7 + c[4] * 20)

        bit_per_byte = expected_bits / original_bytes if original_bytes > 0 else 0
        speed_kb_s = (original_bytes / 1024) / total_time if total_time > 0 else 0

        print(f"\n" + "=" * 40)
        print(f"压缩统计报告 (Time used: {total_time:.2f}s)")
        print(f"Encode Speed (tokens/s): {len(processed_tokens) / total_time:.1f} tokens/s")
        print(f"Encode Speed: {speed_kb_s:.2f} KB/s")
        print(f"-" * 40)
        print(f"1. 命中率:")
        for i in range(1, 5):
            print(f"   - Index {i}: {c[i] / total_tokens:>6.2%} ({c[i]} tokens)")
        print(f"2. 效率与显存指标:")
        print(f"   - 原始文本: {original_bits:>12,} bits")
        print(f"   - 预估压缩: {expected_bits:>12,} bits")
        print(f"   - 理论压缩率: {original_bits / expected_bits:>10.4f}x")
        print(f"   - 绝对压缩率: {bit_per_byte:>10.3f} bit/byte")
        print(f"   - Peak VRAM: {peak_vram:>10.3f} GB")
        print(f"=" * 40)


# ==========================================
# 解码器类 (Decoder)
# ==========================================
class FastRankDecoder:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.args = args

    def decode_file(self, input_compressed_path, output_text_path):
        t0 = time.perf_counter()
        print(f"\n--- Start decoding: {input_compressed_path} ---")

        with open(input_compressed_path, 'r') as f:
            header = f.readline().split()
            batch_size, chunk_length = int(header[0]), int(header[1])
            first_tokens = list(map(int, f.readline().split()))

            decoded_ids = torch.zeros((batch_size, chunk_length), dtype=torch.long, device=self.device)
            decoded_ids[:, 0] = torch.tensor(first_tokens, device=self.device)

            input_ids = decoded_ids[:, 0].unsqueeze(1)
            past_key_values = None

            # --- 执行 warmup 并重置 peak VRAM ---
            warmup_and_reset_peak_memory(self.model, input_ids, self.device)

            # --- 初始化计时器 ---
            time_gpu_infer = 0.0
            time_gpu_topk = 0.0
            time_cpu_decode_and_copy = 0.0

            with torch.inference_mode():
                for step in range(chunk_length - 1):
                    position_ids = torch.full((batch_size, 1), step, dtype=torch.long, device=self.device)

                    if past_key_values is not None:
                        curr_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else \
                            past_key_values[0][0].size(2)
                        if curr_len >= self.args.context_length:
                            past_key_values = slice_kv_cache(past_key_values, self.args.window_length)

                    # 1. 测 GPU Inference Time
                    torch.cuda.synchronize()
                    t_infer_start = time.perf_counter()

                    outputs = self.model(input_ids=input_ids, past_key_values=past_key_values,
                                         position_ids=position_ids, use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    torch.cuda.synchronize()
                    time_gpu_infer += (time.perf_counter() - t_infer_start)

                    # 2. 测 GPU TopK 时间
                    t_topk_start = time.perf_counter()
                    k = 32  # 根据实际需要的最大 rank 设置
                    topk_indices = torch.topk(logits, k=k, dim=-1).indices
                    torch.cuda.synchronize()
                    time_gpu_topk += (time.perf_counter() - t_topk_start)

                    # 3. 测 CPU 文件读取 + 解析 + GPU隐式拷贝耗时
                    t_cpu_start = time.perf_counter()
                    line = f.readline().split()
                    next_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                    for b in range(batch_size):
                        part = line[b].split(',')
                        idx = int(part[0])

                        if idx == 1:
                            # 选择 top-1（最高概率）
                            next_tokens[b] = topk_indices[b, 0]
                        elif idx in [2, 3]:
                            rank = int(part[1])
                            next_tokens[b] = topk_indices[b, rank - 1]
                        else:  # idx == 0 或其他情况
                            next_tokens[b] = int(part[1])

                    # 同步，确保循环里的隐式拷贝全都完成
                    torch.cuda.synchronize()
                    time_cpu_decode_and_copy += (time.perf_counter() - t_cpu_start)

                    decoded_ids[:, step + 1] = next_tokens
                    input_ids = next_tokens.unsqueeze(1)
                    if step % 100 == 0: print(f"\rDecoding {step}/{chunk_length - 1}", end="")

        total_time = time.perf_counter() - t0

        # 记录 Decode Peak VRAM
        peak_vram_gb = get_peak_reserved_gb(self.device)

        # 整理输出：按 Batch 顺序还原并计算解码出的总 Bytes
        final_token_list = decoded_ids.view(-1).tolist()
        decoded_text = self.tokenizer.decode(final_token_list, skip_special_tokens=True)
        decoded_bytes = len(decoded_text.encode('utf-8'))
        speed_kb_s = (decoded_bytes / 1024) / total_time if total_time > 0 else 0

        with open(output_text_path, 'w', encoding='utf-8') as f_out:
            f_out.write(decoded_text)

        # ---print results---
        print(f"\n\n[Decoder 耗时与显存分解]")
        print(f"  -> GPU 模型推理耗时: {time_gpu_infer:.3f} s ({time_gpu_infer / total_time:.1%})")
        print(f"  -> GPU TopK计算耗时: {time_gpu_topk:.3f} s ({time_gpu_topk / total_time:.1%})")
        print(
            f"  -> CPU 解析与隐式拷贝耗时: {time_cpu_decode_and_copy:.3f} s ({time_cpu_decode_and_copy / total_time:.1%})")
        print(f"  -> Decode Peak VRAM: {peak_vram_gb:.3f} GB")
        print(
            f"解码耗时: {total_time:.2f}s | 速度: {(batch_size * chunk_length) / total_time:.1f} tokens/s ({speed_kb_s:.2f} KB/s)")

        return final_token_list  # 返回 Token ID 列表用于验证


# ==========================================
# Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Rank Compression and Decompression Tool")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Pretrained model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA fine-tuned weights (optional)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--window_length", type=int, default=512, help="Sliding window length")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file to compress")
    parser.add_argument("--compressed_file", type=str, default="rankcoding_compressed_data.txt", help="Path to save the compressed binary file")
    parser.add_argument("--output_file", type=str, default="rankcoding_decoded_text.txt", help="Path to save the reconstructed text file")
    
    # Precision control parameter
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Model loading precision")
    args = parser.parse_args()

    # Verify input file existence
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Map precision string to torch dtype
    dtype_mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    model_dtype = dtype_mapping[args.dtype]

    print(f"Loading base model: {args.model_name} on {device} with precision {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map=device
    )

    # Load LoRA weights if specified
    if args.lora_path:
        print(f"Loading LoRA weights from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=model_dtype)

    model.eval()

    # Instantiate Encoder and Decoder
    encoder = FastRankEncoder(model, tokenizer, device, args)
    decoder = FastRankDecoder(model, tokenizer, device, args)

    # 1. Run encoding (returns processed original tokens)
    original_processed_tokens = encoder.encode_file(args.input_file, args.compressed_file)

    # 2. Run decoding (returns reconstructed token IDs)
    decoded_token_ids = decoder.decode_file(args.compressed_file, args.output_file)

    # 3. Lossless verification
    is_lossless = (original_processed_tokens == decoded_token_ids)
    print(f"\n" + "*" * 30)
    print(f"Lossless Verification: {'PASSED (Lossless)' if is_lossless else 'FAILED (Mismatch)'}")
    if not is_lossless:
        print(f"Length mismatch: Original {len(original_processed_tokens)} vs Decoded {len(decoded_token_ids)}")
    print("*" * 30)


if __name__ == "__main__":
    main()
