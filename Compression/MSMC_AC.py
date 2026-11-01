"""
MSMC + Arithmetic Coding（新 head 约定 + teacher-forced encode + round-merged decode）
=====================================================================================
依赖：同目录下的 medusa_rank_compress.py（复用 head 定义、backbone 前向、KV 调度、
batch-invariant 补丁、无损验证），以及 batch_invariant_attention.py（如果开 FP16 补丁）。

与旧版 Medusa+AC 脚本相比：
  1. 新 head 约定：每轮 n 个位置 = [backbone lm_head(+1), head0(+2), head1(+3), ...]；
     --round_size 1 时就是“backbone + AC”（即 AC baseline 的同一套 coder）。
  2. 调度与 Rank 版完全一致：encode 多轮一次性 teacher forcing，decode 每轮 1 次 backbone 调用。
  3. 不再把整张 CDF [B, V+1]（~155MB/位置）搬到 CPU：
       encode：GPU 上 gather 出每条流的 [low, high]，每个子批只搬一次 [items, B, 2]；
       decode：CPU 算出每条流的 target value（B 个整数）→ GPU 上 searchsorted 找 symbol
               → 只搬回 [B, 3]；解出的 token 直接留在 GPU 上。
  4. 频率量化：freq = 1 + floor(p * (T - V))，余数加到 argmax，保证总和恰好为 T，
     默认 T = 2^24（旧版 2^20 + clamp(min=1) 会让 152K 个最小频率吃掉 ~15% 概率质量）。
  5. 压缩文件是二进制：一行 JSON 头 + 每条流 packbits 后的字节。
  6. 统计：每个位置的理想 bits（-log2 p）、量化后 bits、实际 AC bits；
     以及 CPU 端 AC 时间占比（--profile 时额外同步测 backbone 时间），用于 Amdahl 分析。

数值一致性：AC 要求编解码两端的量化 CDF 逐位一致（比 Rank-Tier 只要求 top-32 排序一致更严格）。
encode 端 head/lm_head 的输入形状是 [B, n_rounds, d]，decode 端是 [B, 1, d]，
所以 FP16 下需要 --fixed_tile_attention --batch_invariant_linear；softmax 两边都在 [B, V] 上做。
"""

import os
import json
import math
import time
import queue
import threading
import argparse
import contextlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import MSMC_RC as MC
from MSMC_RC import (
    MedusaHeadCollection, backbone_forward, slice_kv_cache, get_kv_len,
    plan_encode_chunks, DTYPE_MAP, apply_precision_guards, register_attention_patch,
    warmup_and_reset_peak_memory, get_peak_reserved_gb, position_labels, verify, sha256_bytes,
    load_tokenizer, ids_to_bytes, read_input, split_prefix_tail,
    _HAS_BATCH_INVARIANT, _BATCH_INVARIANT_IMPORT_ERROR,
)
if _HAS_BATCH_INVARIANT:
    from batch_invariant_attention import (
        fp32_batch_invariant_attention_forward, fixed_tile_attention_forward,
        batch_invariant_linear_mode,
    )


# ==========================================
# 32-bit Arithmetic Coder（与旧版同一算法，接口改成直接收 [low, high, total]）
# ==========================================

_STATE = 32
_MASK = (1 << _STATE) - 1
_TOP = 1 << (_STATE - 1)
_SECOND = _TOP >> 1
_HALF_MASK = _MASK >> 1
_MIN_RANGE = ((1 << _STATE) >> 2) + 2   # total 必须 <= 这个值


class ACEncoder:
    __slots__ = ("low", "high", "num_underflow", "bits")

    def __init__(self):
        self.low, self.high, self.num_underflow = 0, _MASK, 0
        self.bits = []

    def encode(self, symlow, symhigh, total):
        low, high = self.low, self.high
        rng = high - low + 1
        high = low + symhigh * rng // total - 1
        low = low + symlow * rng // total
        bits = self.bits
        while ((low ^ high) & _TOP) == 0:
            bit = low >> (_STATE - 1)
            bits.append(bit)
            if self.num_underflow:
                bits.extend([bit ^ 1] * self.num_underflow)
                self.num_underflow = 0
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
        while (low & ~high & _SECOND) != 0:
            self.num_underflow += 1
            low = (low << 1) & _HALF_MASK
            high = ((high << 1) & _HALF_MASK) | _TOP | 1
        self.low, self.high = low, high

    def finish(self):
        self.bits.append(1)


class ACDecoder:
    __slots__ = ("low", "high", "code", "bits", "idx", "n")

    def __init__(self, bits):
        self.bits, self.idx, self.n = bits, 0, len(bits)
        self.low, self.high, self.code = 0, _MASK, 0
        for _ in range(_STATE):
            self.code = (self.code << 1) | self._read()

    def _read(self):
        if self.idx < self.n:
            b = self.bits[self.idx]
            self.idx += 1
            return b
        return 0

    def value(self, total):
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // rng

    def update(self, symlow, symhigh, total):
        low, high = self.low, self.high
        rng = high - low + 1
        high = low + symhigh * rng // total - 1
        low = low + symlow * rng // total
        while ((low ^ high) & _TOP) == 0:
            self.code = ((self.code << 1) & _MASK) | self._read()
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
        while (low & ~high & _SECOND) != 0:
            self.code = (self.code & _TOP) | ((self.code << 1) & _HALF_MASK) | self._read()
            low = (low << 1) & _HALF_MASK
            high = ((high << 1) & _HALF_MASK) | _TOP | 1
        self.low, self.high = low, high


# ==========================================
# GPU 端：logits -> 量化频率 / 累积频率（编解码共用，必须完全一致）
# ==========================================

def quantize_freq(logits_2d: torch.Tensor, total: int):
    """
    logits_2d: [B, V]（contiguous）
    返回 freq [B, V] int64（每个 >=1，行和恰好 = total），cum [B, V] 为 inclusive cumsum。
    """
    p = torch.softmax(logits_2d.float(), dim=-1)
    V = p.size(-1)
    freq = torch.floor(p * float(total - V)).to(torch.int64) + 1
    rem = total - freq.sum(dim=-1, keepdim=True)          # 通常 >= 0；浮点误差时可能是很小的负数
    am = p.argmax(dim=-1, keepdim=True)
    freq.scatter_add_(1, am, rem)                          # argmax 频率 >= (total-V)/V >> |rem|
    cum = freq.cumsum(dim=-1)
    return freq, cum


# ==========================================
# Encoder：多轮一次性 teacher forcing（调度与 Rank 版相同）
# ==========================================

class MedusaACEncoder:
    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model, self.lm_head = model, model.lm_head
        self.medusa_heads, self.tokenizer, self.device, self.args = medusa_heads, tokenizer, device, args
        self.n = args.round_size
        self.T = 1 << args.total_bits
        self.q = queue.Queue(maxsize=64)
        self.cpu_ac_time = 0.0

    def _ac_worker(self, encoders):
        T = self.T
        while True:
            item = self.q.get()
            if item is None:
                break
            t = time.perf_counter()
            for lohi in item:                 # item: [n_items, B, 2] 的 list 形式，按 (轮, 位置) 顺序
                for enc, (lo, hi) in zip(encoders, lohi):
                    enc.encode(lo, hi, T)
            self.cpu_ac_time += time.perf_counter() - t

    def encode_file(self, input_file_path, output_path):
        n, T, args = self.n, self.T, self.args
        print(f"\n--- MSMC + AC Encoding (n={n}, T=2^{args.total_bits}, "
              f"encode_rounds_per_batch={args.encode_rounds_per_batch}) ---")
        print("各位置预测器: " + ", ".join(position_labels(self.medusa_heads, n)))

        raw, text = read_input(input_file_path)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        B = args.batch_size
        chunk_length = len(text_tokens) // B
        chunk_length = ((chunk_length - 1) // n) * n + 1
        text_tokens = text_tokens[:chunk_length * B]
        tail = split_prefix_tail(self.tokenizer, raw, text_tokens)   # 原样存入压缩文件
        batches = torch.tensor(text_tokens, device=self.device).view(B, chunk_length)
        first_tokens = batches[:, 0].tolist()
        num_real_steps = chunk_length - 1
        num_rounds = num_real_steps // n

        encoders = [ACEncoder() for _ in range(B)]
        worker = threading.Thread(target=self._ac_worker, args=(encoders,))
        worker.start()

        warmup_and_reset_peak_memory(self.model, self.medusa_heads, batches[:, :1], self.device, n)

        ideal_bits = torch.zeros(n, dtype=torch.float64, device=self.device)
        quant_bits = torch.zeros(n, dtype=torch.float64, device=self.device)
        inv_ln2 = 1.0 / math.log(2)
        log2T = float(args.total_bits)

        chunks = plan_encode_chunks(num_rounds, n, args.context_length, args.window_length,
                                    max(1, args.encode_rounds_per_batch))
        past_key_values = None
        rounds_done = 0
        t0 = time.perf_counter()

        with torch.inference_mode():
            for (s, e, slice_before) in chunks:
                if slice_before:   # 与 decoder 在同一轮、对同样的 cache 内容做截断
                    past_key_values = slice_kv_cache(past_key_values, args.window_length)
                lo = 0 if s == 0 else (s - 1) * n + 1
                hi = (e - 1) * n
                pos_ids = torch.arange(lo, hi + 1, device=self.device).unsqueeze(0).expand(B, -1)
                hs, past_key_values = backbone_forward(
                    self.model, batches[:, lo: hi + 1], past_key_values, pos_ids)

                preds = torch.stack([hs[:, r * n - lo, :] for r in range(s, e)], dim=1)  # [B, R, d]
                logits_list = self.medusa_heads.predict(preds, n, self.lm_head)  # n × [B, R, V]

                R = e - s
                lohi = torch.empty((R * n, B, 2), dtype=torch.int64, device=self.device)
                k = 0
                for lr in range(R):
                    base = (s + lr) * n
                    for j in range(n):
                        tgt = batches[:, base + 1 + j].unsqueeze(1)          # [B, 1]
                        lg = logits_list[j][:, lr, :].contiguous()           # [B, V]，与 decode 同形状
                        freq, cum = quantize_freq(lg, T)
                        hi_ = cum.gather(1, tgt)
                        fq = freq.gather(1, tgt)
                        lohi[k, :, 0:1] = hi_ - fq
                        lohi[k, :, 1:2] = hi_
                        k += 1
                        # 统计（不影响码流）
                        lgf = lg.float()
                        nll = torch.logsumexp(lgf, -1) - lgf.gather(1, tgt).squeeze(1)
                        ideal_bits[j] += (nll.double() * inv_ln2).sum()
                        quant_bits[j] += (log2T - torch.log2(fq.double())).sum()
                self.q.put(lohi.cpu().tolist())                # 每个 chunk 只同步/搬运一次

                rounds_done += R
                el = time.perf_counter() - t0
                print(f"\rEncoding rounds {rounds_done}/{num_rounds} | "
                      f"{rounds_done * n * B / max(el, 1e-9):.0f} tok/s | AC queue={self.q.qsize()}",
                      end="")

        self.q.put(None)
        worker.join()
        for enc in encoders:
            enc.finish()
        total_time = time.perf_counter() - t0
        peak = get_peak_reserved_gb(self.device)

        # 写文件：JSON 头 + 每条流 packbits
        bit_lengths = [len(e.bits) for e in encoders]
        payloads = [np.packbits(np.array(e.bits, dtype=np.uint8)).tobytes() for e in encoders]
        header = dict(batch_size=B, num_real_steps=num_real_steps, window_length=args.window_length,
                      round_size=n, head_offset_start=self.medusa_heads.head_offset_start,
                      total_bits=args.total_bits, first_tokens=first_tokens, bit_lengths=bit_lengths,
                      tail_len=len(tail), kv_sink=MC.KV_SINK)
        with open(output_path, 'wb') as f:
            f.write(json.dumps(header).encode() + b"\n")
            for p in payloads:
                f.write(p)
            f.write(tail)

        # 统计（分母 = 整个原文件；文件总大小含头部和原样尾部）
        orig_bytes = len(raw)
        payload_bits = sum(bit_lengths)
        file_bytes = os.path.getsize(output_path)
        ib, qb = ideal_bits.tolist(), quant_bits.tolist()
        n_sym_per_pos = num_rounds * B
        print(f"\n\n{'=' * 60}")
        print(f"MSMC + AC 统计 (encode 耗时 {total_time:.2f}s, 其中 CPU AC {self.cpu_ac_time:.2f}s"
              f"，与 GPU 重叠执行)")
        print(f"{'=' * 60}")
        print(f"round_size n = {n} | encode backbone 调用: {len(chunks)} 次 "
              f"(其中 {sum(1 for c in chunks if c[2])} 次前置 KV 截断)")
        print(f"Encode 速度: {orig_bytes / 1024 / total_time:.2f} KB/s "
              f"({len(text_tokens) / total_time:.0f} tok/s)")
        print(f"{'-' * 60}")
        print("按位置 bits/token:  理想(-log2 p) | 量化后 | ")
        for label, a, b in zip(position_labels(self.medusa_heads, n), ib, qb):
            print(f"  {label:12s}: {a / n_sym_per_pos:7.3f} | {b / n_sym_per_pos:7.3f}")
        print(f"{'-' * 60}")
        print(f"原始大小      : {orig_bytes:,} bytes")
        print(f"AC payload    : {payload_bits:,} bits -> {payload_bits / orig_bytes:.4f} bpb")
        print(f"文件总大小    : {file_bytes:,} bytes (含头部 + 原样尾部 {len(tail):,} bytes) "
              f"-> {file_bytes * 8 / orig_bytes:.4f} bpb")
        print(f"理想 (-log2 p): {sum(ib) / orig_bytes:.4f} bpb | 量化后: {sum(qb) / orig_bytes:.4f} bpb")
        print(f"AC 开销       : {payload_bits - sum(qb):+,.0f} bits 相对量化熵, "
              f"{payload_bits - sum(ib):+,.0f} bits 相对理想熵")
        if peak is not None:
            print(f"Encode Peak GPU 显存: {peak:.3f} GB")
        print(f"{'=' * 60}")
        return text_tokens


# ==========================================
# Decoder：round-merged（每轮 1 次 backbone 调用）
# ==========================================

class MedusaACDecoder:
    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model, self.lm_head = model, model.lm_head
        self.medusa_heads, self.tokenizer, self.device, self.args = medusa_heads, tokenizer, device, args

    def decode_file(self, compressed_path, output_text_path):
        print(f"\n--- MSMC + AC Decoding: {compressed_path} ---")
        with open(compressed_path, 'rb') as f:
            header = json.loads(f.readline().decode())
            data = f.read()

        B = header["batch_size"]
        num_real_steps = header["num_real_steps"]
        W = header["window_length"]
        n = header["round_size"]
        T = 1 << header["total_bits"]
        if header["head_offset_start"] != self.medusa_heads.head_offset_start:
            raise ValueError("压缩文件与当前 heads 的位置约定不一致")
        if n > self.medusa_heads.max_round_size:
            raise ValueError(f"压缩文件 round_size={n} 超过 heads 支持的 {self.medusa_heads.max_round_size}")

        decoders, off = [], 0
        for nbits in header["bit_lengths"]:
            nbytes = (nbits + 7) // 8
            bits = np.unpackbits(np.frombuffer(data[off: off + nbytes], dtype=np.uint8))[:nbits]
            decoders.append(ACDecoder(bits.tolist()))
            off += nbytes
        tail = data[off: off + header.get("tail_len", 0)]
        MC.KV_SINK = header.get("kv_sink", 0)              # 与编码端保持一致

        chunk_length = num_real_steps + 1
        decoded = torch.zeros((B, chunk_length), dtype=torch.long, device=self.device)
        decoded[:, 0] = torch.tensor(header["first_tokens"], device=self.device)
        round_input = decoded[:, 0:1]
        round_start_pos, past_key_values = 0, None
        MAX_L = self.args.context_length

        warmup_and_reset_peak_memory(self.model, self.medusa_heads, round_input, self.device, n)
        profile = self.args.profile
        t_backbone = t_cpu_ac = 0.0
        step = 0
        t0 = time.perf_counter()

        with torch.inference_mode():
            while step < chunk_length - 1:
                cur = min(n, chunk_length - 1 - step)
                if past_key_values is not None and get_kv_len(past_key_values) >= MAX_L:
                    past_key_values = slice_kv_cache(past_key_values, W)
                pos_ids = torch.arange(round_start_pos, round_start_pos + round_input.size(1),
                                       device=self.device).unsqueeze(0).expand(B, -1)

                tb = time.perf_counter()
                hs, past_key_values = backbone_forward(self.model, round_input, past_key_values, pos_ids)
                logits_list = self.medusa_heads.predict(hs[:, -1:, :], n, self.lm_head)
                if profile:
                    torch.cuda.synchronize()
                    t_backbone += time.perf_counter() - tb

                toks = []
                for j in range(cur):
                    freq, cum = quantize_freq(logits_list[j][:, -1, :].contiguous(), T)

                    tc = time.perf_counter()
                    vals = [d.value(T) for d in decoders]
                    t_cpu_ac += time.perf_counter() - tc

                    v = torch.tensor(vals, dtype=torch.int64).to(self.device, non_blocking=True)
                    sym = torch.searchsorted(cum, v.unsqueeze(1), right=True)   # [B, 1]
                    hi = cum.gather(1, sym)
                    lo = hi - freq.gather(1, sym)
                    lohi = torch.cat([lo, hi], dim=1).cpu().tolist()              # 只搬 [B, 2]

                    tc = time.perf_counter()
                    for d, (l, h) in zip(decoders, lohi):
                        d.update(l, h, T)
                    t_cpu_ac += time.perf_counter() - tc

                    tok = sym.squeeze(1)                    # token 留在 GPU 上
                    decoded[:, step + 1 + j] = tok
                    toks.append(tok)

                round_input = torch.stack(toks, dim=1)
                round_start_pos = step + 1
                step += cur
                if step % 200 < n:
                    print(f"\rDecoding {step}/{chunk_length - 1}", end="")

        peak = get_peak_reserved_gb(self.device)
        token_list = decoded.view(-1).tolist()
        out_bytes = ids_to_bytes(self.tokenizer, token_list) + tail
        total_time = time.perf_counter() - t0           # 计入字节还原
        with open(output_text_path, 'wb') as f:
            f.write(out_bytes)

        print(f"\n解码耗时: {total_time:.2f}s | {B * chunk_length / total_time:.0f} tok/s | "
              f"{len(out_bytes) / 1024 / total_time:.2f} KB/s")
        print(f"  CPU AC 时间: {t_cpu_ac:.2f}s ({t_cpu_ac / total_time:.1%})")
        if profile:
            print(f"  backbone+heads 时间: {t_backbone:.2f}s ({t_backbone / total_time:.1%})；"
                  f"其余（量化/searchsorted/传输/同步）: "
                  f"{total_time - t_backbone - t_cpu_ac:.2f}s")
        if peak is not None:
            print(f"Decode Peak GPU 显存: {peak:.3f} GB")
        return token_list


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lora_path", default="", help="只允许 domain-level LoRA")
    ap.add_argument("--medusa_heads_dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--window_length", type=int, default=512)
    ap.add_argument("--round_size", type=int, default=2, help="每轮位置数 n；n=1 即 backbone+AC")
    ap.add_argument("--total_bits", type=int, default=24, help="量化总频率 T=2^total_bits（<=30）")
    ap.add_argument("--kv_sink", type=int, default=0, help="KV 截断时保留的开头 token 数；0 = 旧行为")
    ap.add_argument("--encode_rounds_per_batch", type=int, default=8)
    ap.add_argument("--mode", default="both", choices=["both", "encode", "decode"])
    ap.add_argument("--input_file", default=None)
    ap.add_argument("--compressed_file", default="output_ac.bin")
    ap.add_argument("--output_file", default="decoded_ac.txt")
    ap.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--batch_invariant_attention", action="store_true")
    ap.add_argument("--fixed_tile_attention", action="store_true")
    ap.add_argument("--batch_invariant_linear", action="store_true")
    ap.add_argument("--profile", action="store_true", help="decode 时同步测 backbone 时间（略慢）")
    args = ap.parse_args()
    MC.KV_SINK = args.kv_sink

    if (1 << args.total_bits) > _MIN_RANGE:
        raise ValueError("total_bits 过大：必须满足 2^total_bits <= 2^30")
    if args.mode in ("both", "encode") and not (args.input_file and os.path.exists(args.input_file)):
        raise FileNotFoundError(f"找不到输入文件: {args.input_file}")
    if (args.batch_invariant_attention or args.fixed_tile_attention or args.batch_invariant_linear) \
            and not _HAS_BATCH_INVARIANT:
        raise ImportError(f"未能导入 batch_invariant_attention.py：{_BATCH_INVARIANT_IMPORT_ERROR!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[args.dtype]
    apply_precision_guards()
    if args.fixed_tile_attention:
        register_attention_patch(fixed_tile_attention_forward, "fixed_tile_attention")
    elif args.batch_invariant_attention:
        register_attention_patch(fp32_batch_invariant_attention_forward, "batch_invariant_attention")

    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device,
                                                 attn_implementation=args.attn_implementation)
    if args.lora_path:
        from peft import PeftModel
        print(f"[Warn] 加载 LoRA {args.lora_path}：请确认它不是在测试文件上训练的")
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=dtype).merge_and_unload()
    model.eval()

    heads = MedusaHeadCollection.load(args.medusa_heads_dir, device, dtype=dtype).eval()
    if not heads.uses_base_for_first:
        print("[Warn] 这是旧约定的 heads（head0 预测 +1），建议用新训练的 heads")
    if not (1 <= args.round_size <= heads.max_round_size):
        raise ValueError(f"round_size 必须在 [1, {heads.max_round_size}]")
    print(f"round_size n = {args.round_size}: " + ", ".join(position_labels(heads, args.round_size)))

    ctx = batch_invariant_linear_mode() if args.batch_invariant_linear else contextlib.nullcontext()
    ref_tokens = None
    with ctx:
        if args.mode in ("both", "encode"):
            ref_tokens = MedusaACEncoder(model, heads, tokenizer, device, args).encode_file(
                args.input_file, args.compressed_file)
            with open(args.compressed_file, 'rb') as f:
                print(f"压缩文件 sha256: {sha256_bytes(f.read())}")
        if args.mode in ("both", "decode"):
            decoded = MedusaACDecoder(model, heads, tokenizer, device, args).decode_file(
                args.compressed_file, args.output_file)

    if args.mode in ("both", "decode"):
        if args.input_file and os.path.exists(args.input_file):
            ok = verify(tokenizer, args.input_file, args.output_file, decoded, ref_tokens)
            raise SystemExit(0 if ok else 1)
        print("（未提供 --input_file，跳过无损验证）")


if __name__ == "__main__":
    main()
