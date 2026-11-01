"""
MSMC Rank -- for domain heads
===============================================================================
用法：
  python medusa_rank_compress.py --model_name Qwen/Qwen2.5-0.5B \
      --medusa_heads_dir ./medusa_domain_wiki/best --round_size 2 \
      --input_file test/wiki_03.txt --compressed_file c.txt --output_file d.txt \
      --dtype fp16 --fixed_tile_attention --batch_invariant_linear

  # reload 测试：两个独立进程
  python medusa_rank_compress.py ... --mode encode --input_file X --compressed_file c.txt
  python medusa_rank_compress.py ... --mode decode --compressed_file c.txt \
      --output_file d.txt --input_file X        # input_file 只用来做逐字节比对
"""

import os
import json
import time
import queue
import hashlib
import threading
import argparse
import contextlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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


# ==========================================
# 字节级无损 I/O
#   - tokenizer 关掉 normalizer（Qwen2 默认 NFC），保证 token 序列可逆回原始字节
#   - token -> 字节 用 byte-level BPE 的逆映射，而不是 tokenizer.decode
#     （decode 遇到被截断的多字节 UTF-8 字符会输出 U+FFFD，导致字节不一致）
#   - 装不进 B 条流的尾部原样存储（计入压缩大小）
# ==========================================

try:
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
except ImportError:  # 极老版本兜底
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
            + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

_BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        from tokenizers import normalizers
        if tok.backend_tokenizer.normalizer is not None:
            tok.backend_tokenizer.normalizer = normalizers.Sequence([])
    except Exception as e:
        print(f"[Warn] 无法关闭 tokenizer normalizer：{e}（会在编码时检查字节可逆性）")
    return tok


def ids_to_bytes(tokenizer, ids):
    out = bytearray()
    for t in tokenizer.convert_ids_to_tokens(ids):
        try:
            out.extend(_BYTE_DECODER[c] for c in t)
        except KeyError:            # 特殊 token 等非 byte-level 映射的情况
            out.extend(t.encode("utf-8"))
    return bytes(out)


def read_input(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw, raw.decode("utf-8")   # 不用文本模式读，避免 \r\n 被改写


def split_prefix_tail(tokenizer, raw, processed_ids):
    prefix = ids_to_bytes(tokenizer, processed_ids)
    if not raw.startswith(prefix):
        raise RuntimeError("tokenizer 不是字节可逆的：token 序列还原出的字节不是原文前缀。"
                           "检查 normalizer / 特殊 token 设置。")
    return raw[len(prefix):]


# ==========================================
# Medusa Head 结构（与 train_medusa_domain.py 完全一致）
# ==========================================

class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear(hidden_states)) + hidden_states
        return self.lm_head(x)


class MedusaHeadCollection(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int,
                 head_offset_start: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_offset_start = head_offset_start
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size) for _ in range(num_heads)
        ])

    @property
    def uses_base_for_first(self) -> bool:
        return self.head_offset_start == 2

    @property
    def max_round_size(self) -> int:
        return self.num_heads + (1 if self.uses_base_for_first else 0)

    def predict(self, hidden_states: torch.Tensor, round_size: int, base_lm_head: nn.Module):
        """
        hidden_states: [B, hidden] 或 [B, L, hidden]（L 个 predictor 位置一起算）
        返回长度为 round_size 的 list，第 j 个元素预测本轮第 j+1 个 token。
          新约定：[base_lm_head, head0, head1, ...]
          旧约定：[head0, head1, ...]
        """
        if not (1 <= round_size <= self.max_round_size):
            raise ValueError(f"round_size={round_size} 超出范围 [1, {self.max_round_size}]")
        if self.uses_base_for_first:
            outs = [base_lm_head(hidden_states)]
            outs += [h(hidden_states) for h in list(self.heads)[: round_size - 1]]
        else:
            outs = [h(hidden_states) for h in list(self.heads)[:round_size]]
        return outs

    @classmethod
    def load(cls, save_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
        with open(os.path.join(save_dir, "medusa_config.json")) as f:
            config = json.load(f)
        heads = cls(config["hidden_size"], config["vocab_size"], config["num_heads"],
                    head_offset_start=config.get("head_offset_start", 1))
        state = torch.load(os.path.join(save_dir, "medusa_heads.pt"), map_location=device)
        heads.load_state_dict(state)
        return heads.to(device=device, dtype=dtype)


# ==========================================
# Backbone 前向：只取 normed hidden，不算整张 logits
# ==========================================

def backbone_forward(model, input_ids, past_key_values, position_ids):
    """返回 (last_hidden_state[B, L, d]（已过 final norm）, past_key_values)。"""
    out = model.model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
    )
    return out.last_hidden_state, out.past_key_values


# ==========================================
# KV Cache 工具（与旧版相同）
# ==========================================

# 截断 KV cache 时保留的开头 token 数（attention sink，见 StreamingLLM）。
# 0 = 旧行为（只保留最后 window_length 个）。编码端写入文件头，解码端从文件头读回，两端一致。
KV_SINK = 0


def _keep(t, window_length, sink):
    if sink <= 0 or t.size(-2) <= window_length:
        return t[..., -window_length:, :]
    return torch.cat([t[..., :sink, :], t[..., -(window_length - sink):, :]], dim=-2)


def slice_kv_cache(past_key_values, window_length, sink=None):
    """截断到 window_length 个条目；sink>0 时保留最前面 sink 个 + 最后 window_length-sink 个。
    截断后总长度仍是 window_length，所以 plan_encode_chunks 的调度推演不变。"""
    sink = KV_SINK if sink is None else sink
    if past_key_values is None:
        return None
    if isinstance(past_key_values, DynamicCache):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = _keep(past_key_values.key_cache[i], window_length, sink)
            past_key_values.value_cache[i] = _keep(past_key_values.value_cache[i], window_length, sink)
        return past_key_values
    return tuple((_keep(k, window_length, sink), _keep(v, window_length, sink))
                 for k, v in past_key_values)


def get_kv_len(past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].size(2)


def plan_encode_chunks(num_rounds, n, context_length, window_length, rounds_per_chunk):
    """
    按 decoder 的实际规则推演 KV cache：decoder 第 r 轮前若 cache 长度 >= context_length，
    就截断到最后 window_length 个；第 r 轮的输入长度是 1（r=0）或 n。
    encoder 的每个 chunk 恰好覆盖 decoder 连续若干轮的输入，且不跨越任何截断点，
    于是每轮 predictor hidden 看到的上下文与 decoder 完全相同。
    返回 [(s, e, slice_before)]：chunk 覆盖第 s..e-1 轮，slice_before 表示 chunk 前先截断。
    """
    slice_before, kv = set(), 0
    for r in range(num_rounds):
        if kv >= context_length:
            slice_before.add(r)
            kv = window_length
        kv += 1 if r == 0 else n
    chunks, s = [], 0
    while s < num_rounds:
        e = s + 1
        while e < num_rounds and e - s < rounds_per_chunk and e not in slice_before:
            e += 1
        chunks.append((s, e, s in slice_before))
        s = e
    return chunks


def compute_round_segments(num_rounds, step_size, context_length, window_length):
    segments = []
    cur_len = 0
    seg_start = 0
    for r in range(num_rounds):
        if cur_len >= context_length:
            segments.append((seg_start, r))
            seg_start = r
            cur_len = window_length
        cur_len += step_size
    segments.append((seg_start, num_rounds))
    return segments


# ==========================================
# GPU 显存 / 精度 工具（与旧版相同）
# ==========================================

DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def _is_cuda(device) -> bool:
    return "cuda" in str(device)


def apply_precision_guards():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def register_attention_patch(attn_fn, label):
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
    print(f"[{label}] 已启用" + ("（整体替换 Qwen2Attention.forward）" if patched_class else "（通过注册表/模块函数）"))


def warmup_and_reset_peak_memory(model, medusa_heads, input_ids, device, round_size):
    if not _is_cuda(device):
        return
    torch.cuda.synchronize()
    with torch.inference_mode():
        B, L = input_ids.shape
        dummy_pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        hs, _ = backbone_forward(model, input_ids, None, dummy_pos)
        _ = medusa_heads.predict(hs[:, -1:, :], round_size, model.lm_head)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_reserved_gb(device):
    if not _is_cuda(device):
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


TIER_NAMES = {1: "rank=1", 2: "rank 2-4", 3: "rank 5-32", 4: "miss(>32)"}
TIER_BITS = {1: 2, 2: 4, 3: 7, 4: 20}   # 与旧版的预估码长一致


def position_labels(medusa_heads, round_size):
    if medusa_heads.uses_base_for_first:
        return ["base(+1)"] + [f"head{k}(+{k + 2})" for k in range(round_size - 1)]
    return [f"head{k}(+{k + 1})" for k in range(round_size)]


# ==========================================
# Encoder：多轮一次性教师强制（调度与旧版相同）
# ==========================================

class MedusaRankEncoder:
    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model = model
        self.lm_head = model.lm_head
        self.medusa_heads = medusa_heads
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.step_size = args.round_size
        self.write_queue = queue.Queue(maxsize=200)
        self.final_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.pos_counts = [{1: 0, 2: 0, 3: 0, 4: 0} for _ in range(self.step_size)]
        self.peak_vram_gb = None

    def _writer_worker(self, output_path, batch_size, num_real_steps, first_tokens):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{batch_size} {num_real_steps} {self.args.window_length} {self.step_size} "
                    f"{self.medusa_heads.head_offset_start} {KV_SINK}\n")
            f.write(" ".join(map(str, first_tokens)) + "\n")

            while True:
                item = self.write_queue.get()
                if item is None:
                    break
                for pos_offset, (ranks_cpu, targets_cpu) in enumerate(item):
                    step_data = []
                    pc = self.pos_counts[pos_offset]
                    for b in range(batch_size):
                        r = int(ranks_cpu[b])
                        if r == 1:
                            step_data.append("1"); tier = 1
                        elif 2 <= r <= 4:
                            step_data.append(f"2,{r}"); tier = 2
                        elif 5 <= r <= 32:
                            step_data.append(f"3,{r}"); tier = 3
                        else:
                            step_data.append(f"4,{int(targets_cpu[b])}"); tier = 4
                        self.final_counts[tier] += 1
                        pc[tier] += 1
                    f.write(" ".join(step_data) + "\n")
                self.write_queue.task_done()

    def encode_file(self, input_file_path, output_compressed_path):
        print(f"\n--- MSMC Rank Encoding (round_size n={self.step_size}, "
              f"trained heads={self.medusa_heads.num_heads}, "
              f"head_offset_start={self.medusa_heads.head_offset_start}, "
              f"encode_rounds_per_batch={self.args.encode_rounds_per_batch}) ---")
        print("本轮各位置使用的预测器: " + ", ".join(position_labels(self.medusa_heads, self.step_size)))

        raw, text = read_input(input_file_path)
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)

        B = self.args.batch_size
        chunk_length = len(text_tokens) // B
        usable_len = ((chunk_length - 1) // self.step_size) * self.step_size + 1
        chunk_length = usable_len
        valid_token_len = chunk_length * B
        text_tokens = text_tokens[:valid_token_len]
        self.raw_len = len(raw)
        self.tail = split_prefix_tail(self.tokenizer, raw, text_tokens)
        with open(output_compressed_path + ".tail", "wb") as f:   # 尾部原样存储
            f.write(self.tail)

        batches = torch.tensor(text_tokens, device=self.device).view(B, chunk_length)
        first_tokens = batches[:, 0].cpu().numpy()
        num_real_steps = chunk_length - 1
        num_rounds = num_real_steps // self.step_size

        writer_thread = threading.Thread(
            target=self._writer_worker,
            args=(output_compressed_path, B, num_real_steps, first_tokens)
        )
        writer_thread.start()

        warmup_and_reset_peak_memory(self.model, self.medusa_heads, batches[:, :1],
                                     self.device, self.step_size)

        n = self.step_size
        chunks = plan_encode_chunks(num_rounds, n, self.args.context_length,
                                    self.args.window_length,
                                    max(1, self.args.encode_rounds_per_batch))

        past_key_values = None
        t0 = time.perf_counter()
        rounds_done = 0

        with torch.inference_mode():
            for (s, e, slice_before) in chunks:
                if slice_before:   # 与 decoder 在同一轮、对同样的 cache 内容做截断
                    past_key_values = slice_kv_cache(past_key_values, self.args.window_length)

                # 本 chunk = decoder 第 s..e-1 轮的输入拼在一起；每轮 predictor 位置都在 chunk 内
                lo = 0 if s == 0 else (s - 1) * n + 1
                hi = (e - 1) * n
                chunk_input_ids = batches[:, lo: hi + 1]
                position_ids = torch.arange(lo, hi + 1, device=self.device).unsqueeze(0).expand(B, -1)

                hs, past_key_values = backbone_forward(
                    self.model, chunk_input_ids, past_key_values, position_ids
                )
                predictor_states = torch.stack(
                    [hs[:, r * n - lo, :] for r in range(s, e)], dim=1)   # [B, n_rounds, d]

                logits_list = self.medusa_heads.predict(
                    predictor_states, n, self.lm_head
                )  # 每个元素 [B, n_rounds, V]

                for local_r in range(e - s):
                    r = s + local_r
                    round_start_pos = r * n
                    batch_write_items = []
                    for pos_offset in range(n):
                        target_tokens = batches[:, round_start_pos + 1 + pos_offset]
                        logits_this = logits_list[pos_offset][:, local_r, :]
                        _, sorted_indices = torch.topk(logits_this, k=32, dim=-1)
                        matches = (sorted_indices == target_tokens.unsqueeze(1))
                        ranks = matches.int().argmax(dim=-1) + 1
                        ranks[~matches.any(dim=-1)] = 33
                        batch_write_items.append(
                            (ranks.cpu().numpy(), target_tokens.cpu().numpy())
                        )
                    self.write_queue.put(batch_write_items)

                rounds_done += e - s
                elapsed = time.perf_counter() - t0
                speed = rounds_done * n * B / elapsed if elapsed > 0 else 0
                print(f"\rEncoding rounds {rounds_done}/{num_rounds} | "
                      f"speed={speed:.0f} tok/s | Queue={self.write_queue.qsize()}", end="")

        self.write_queue.put(None)
        writer_thread.join()
        total_time = time.perf_counter() - t0
        self.peak_vram_gb = get_peak_reserved_gb(self.device)
        self._print_stats(text_tokens, total_time)
        return text_tokens

    def _print_stats(self, processed_tokens, total_time):
        original_bytes = self.raw_len                 # 整个原文件（含未进流的尾部）
        original_bits = original_bytes * 8

        c = self.final_counts
        total_tokens = sum(c.values())
        tail_bits = 8 * len(self.tail)
        expected_bits = sum(c[t] * TIER_BITS[t] for t in c) + tail_bits
        bits_per_byte = expected_bits / original_bytes if original_bytes > 0 else float('nan')
        encode_speed_kbps = (original_bytes / 1024) / total_time if total_time > 0 else float('nan')

        print(f"\n\n{'=' * 55}")
        print(f"MSMC Rank Compression 统计 (耗时: {total_time:.2f}s)")
        print(f"{'=' * 55}")
        print(f"round_size n = {self.step_size}  (训练 head 总数 {self.medusa_heads.num_heads})")
        print(f"每次 backbone 前向最多处理 {self.args.encode_rounds_per_batch} 轮"
              f"（{self.args.encode_rounds_per_batch * self.step_size} 个位置）")
        print(f"吞吐量: {len(processed_tokens) / total_time:.1f} tokens/s")
        print(f"Encode 速度: {encode_speed_kbps:.2f} KB/s")
        print(f"{'-' * 55}")
        print("命中率（全部位置聚合）:")
        for i in range(1, 5):
            print(f"  {TIER_NAMES[i]:10s}: {c[i] / total_tokens:>6.2%} ({c[i]} tokens)")
        print(f"{'-' * 55}")
        print("按位置拆分（rank=1 / 2-4 / 5-32 / miss | 预估 bits/token）:")
        for label, pc in zip(position_labels(self.medusa_heads, self.step_size), self.pos_counts):
            n = sum(pc.values())
            if n == 0:
                continue
            bpt = sum(pc[t] * TIER_BITS[t] for t in pc) / n
            print(f"  {label:12s}: " + " / ".join(f"{pc[t] / n:.3f}" for t in range(1, 5))
                  + f" | {bpt:.2f}")
        print(f"{'-' * 55}")
        print(f"原始文本: {original_bits:>12,} bits ({original_bytes:,} bytes，整个文件)")
        print(f"预估压缩: {expected_bits:>12,} bits（tier 码长估算 + 原样尾部 {len(self.tail):,} bytes）")
        print(f"压缩率(bit/byte): {bits_per_byte:.4f} bit/byte")
        print(f"理论压缩率: {original_bits / expected_bits:>10.4f}x")
        if self.peak_vram_gb is not None:
            print(f"Encode 阶段 Peak GPU 显存 (max_memory_reserved): {self.peak_vram_gb:.3f} GB")
        print(f"{'=' * 55}")


# ==========================================
# Decoder：轮次合并（每轮 1 次 backbone 前向，与旧版相同）
# ==========================================

class MedusaRankDecoder:
    def __init__(self, model, medusa_heads, tokenizer, device, args):
        self.model = model
        self.lm_head = model.lm_head
        self.medusa_heads = medusa_heads
        self.tokenizer = tokenizer
        self.device = device
        self.args = args
        self.peak_vram_gb = None
        self.header = None

    def decode_file(self, input_compressed_path, output_text_path):
        print(f"\n--- MSMC Rank Decoding: {input_compressed_path} ---")

        with open(input_compressed_path, 'r') as f:
            header = list(map(int, f.readline().split()))
            batch_size, num_real_steps, window_length, step_size = header[:4]
            head_offset_start = header[4] if len(header) >= 5 else 1  # 旧格式没有这个字段
            global KV_SINK
            KV_SINK = header[5] if len(header) >= 6 else 0          # 与编码端保持一致
            self.header = dict(batch_size=batch_size, num_real_steps=num_real_steps)
            first_tokens = list(map(int, f.readline().split()))
            all_lines = [line.split() for line in f]

        if head_offset_start != self.medusa_heads.head_offset_start:
            raise ValueError(
                f"压缩文件的 head_offset_start={head_offset_start}，但加载的 heads 是 "
                f"{self.medusa_heads.head_offset_start}：两边的位置约定不一致，无法解码。")
        if step_size > self.medusa_heads.max_round_size:
            raise ValueError(f"压缩文件要求 round_size={step_size}，"
                             f"但当前 heads 最多支持 {self.medusa_heads.max_round_size}。")
        print(f"压缩文件记录的 round_size n = {step_size}")

        chunk_length = num_real_steps + 1
        decoded_ids = torch.zeros((batch_size, chunk_length), dtype=torch.long, device=self.device)
        decoded_ids[:, 0] = torch.tensor(first_tokens, device=self.device)

        round_input_ids = decoded_ids[:, 0].unsqueeze(1)  # [B, 1]，bootstrap
        round_start_pos = 0
        past_key_values = None
        MAX_L = self.args.context_length
        W = window_length

        warmup_and_reset_peak_memory(self.model, self.medusa_heads, round_input_ids,
                                     self.device, step_size)

        step = 0
        line_idx = 0
        t0 = time.perf_counter()

        with torch.inference_mode():
            while step < chunk_length - 1:
                cur_step_size = min(step_size, chunk_length - 1 - step)

                if past_key_values is not None and get_kv_len(past_key_values) >= MAX_L:
                    past_key_values = slice_kv_cache(past_key_values, W)

                round_pos_ids = torch.arange(
                    round_start_pos, round_start_pos + round_input_ids.size(1),
                    dtype=torch.long, device=self.device
                ).unsqueeze(0).expand(batch_size, -1)

                hs, past_key_values = backbone_forward(
                    self.model, round_input_ids, past_key_values, round_pos_ids
                )
                hidden_states = hs[:, -1:, :]  # [B, 1, d]

                logits_list = self.medusa_heads.predict(hidden_states, step_size, self.lm_head)
                all_logits = [lg[:, -1, :] for lg in logits_list]

                decoded_tokens_this_round = []
                for pos_offset in range(cur_step_size):
                    topk_indices = torch.topk(all_logits[pos_offset], k=32, dim=-1).indices

                    line = all_lines[line_idx]
                    line_idx += 1

                    # 向量化：CPU 上只解析字符串，GPU 上一次 gather + where
                    tiers, vals = [], []
                    for s in line:
                        if s == "1":
                            tiers.append(1); vals.append(1)
                        else:
                            t_, v_ = s.split(',')
                            tiers.append(int(t_)); vals.append(int(v_))
                    tier_t = torch.tensor(tiers, device=self.device)
                    val_t = torch.tensor(vals, device=self.device)
                    rank_idx = (val_t - 1).clamp(0, 31).unsqueeze(1)
                    from_topk = topk_indices.gather(1, rank_idx).squeeze(1)
                    next_tokens = torch.where(tier_t == 4, val_t, from_topk)

                    decoded_ids[:, step + 1 + pos_offset] = next_tokens
                    decoded_tokens_this_round.append(next_tokens)

                round_input_ids = torch.stack(decoded_tokens_this_round, dim=1)
                round_start_pos = step + 1
                step += cur_step_size

                if step % 100 < step_size:
                    print(f"\rDecoding {step}/{chunk_length - 1}", end="")

        self.peak_vram_gb = get_peak_reserved_gb(self.device)

        final_token_list = decoded_ids.view(-1).tolist()
        tail_path = input_compressed_path + ".tail"
        tail = open(tail_path, "rb").read() if os.path.exists(tail_path) else b""
        decoded_bytes = ids_to_bytes(self.tokenizer, final_token_list) + tail
        total_time = time.perf_counter() - t0          # 计入字节还原
        decode_speed_kbps = (len(decoded_bytes) / 1024) / total_time if total_time > 0 else float('nan')

        print(f"\n解码耗时: {total_time:.2f}s | "
              f"速度: {(batch_size * chunk_length) / total_time:.1f} tokens/s | "
              f"{decode_speed_kbps:.2f} KB/s")
        if self.peak_vram_gb is not None:
            print(f"Decode 阶段 Peak GPU 显存 (max_memory_reserved): {self.peak_vram_gb:.3f} GB")

        with open(output_text_path, 'wb') as f_out:
            f_out.write(decoded_bytes)
        return final_token_list


# ==========================================
# 无损验证
# ==========================================

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify(tokenizer, input_file, output_file, decoded_tokens, ref_tokens=None):
    """
    两层验证：
      1) token 级：解码 token 序列 == 编码时处理的 token 序列
      2) 文件级：解码输出文件 == 原始文件，逐字节完全相同（含尾部），sha256 一致
    """
    raw, text = read_input(input_file)
    with open(output_file, 'rb') as f:
        out_bytes = f.read()

    if ref_tokens is None:
        ref_tokens = tokenizer.encode(text, add_special_tokens=False)[:len(decoded_tokens)]
    token_ok = (ref_tokens == decoded_tokens)
    file_ok = (out_bytes == raw)

    print(f"\n{'*' * 50}")
    print(f"Token 级:  {'通过' if token_ok else '失败'}")
    print(f"文件级:    {'通过' if file_ok else '失败'}  "
          f"(sha256 out={sha256_bytes(out_bytes)[:16]}… orig={sha256_bytes(raw)[:16]}…, "
          f"{len(out_bytes):,}/{len(raw):,} bytes)")
    if not file_ok and token_ok:
        n = min(len(out_bytes), len(raw))
        i = next((k for k in range(n) if out_bytes[k] != raw[k]), n)
        print(f"首个不同字节位置: {i}")
    if not token_ok:
        n = min(len(ref_tokens), len(decoded_tokens))
        for i in range(n):
            if ref_tokens[i] != decoded_tokens[i]:
                print(f"首次 token 不匹配位置: {i} | 原值 {ref_tokens[i]} | 解码值 {decoded_tokens[i]}")
                break
        if len(ref_tokens) != len(decoded_tokens):
            print(f"长度不一致: 原始 {len(ref_tokens)} vs 解码 {len(decoded_tokens)}")
    ok = token_ok and file_ok
    print(f"无损验证: {'通过 (Lossless)' if ok else '失败 (Mismatch)'}")
    print('*' * 50)
    return ok


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lora_path", type=str, default="",
                        help="只允许 domain-level LoRA；per-file LoRA 需要计入压缩大小")
    parser.add_argument("--medusa_heads_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--window_length", type=int, default=512)
    parser.add_argument("--kv_sink", type=int, default=0,
                        help="KV 截断时保留的开头 token 数（attention sink）；0 = 旧行为")
    parser.add_argument("--round_size", type=int, default=None,
                        help="每轮位置数 n（论文里的 n）。新 heads: 1..K+1，n=1 等价 Rank Coding")
    parser.add_argument("--num_active_heads", type=int, default=None,
                        help="（兼容旧参数）使用的 Medusa head 个数；新 heads 下 n = 该值 + 1")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "encode", "decode"])
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--compressed_file", type=str, default="output.txt")
    parser.add_argument("--output_file", type=str, default="decoded_output.txt")
    parser.add_argument("--encode_rounds_per_batch", type=int, default=8)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--batch_invariant_attention", action="store_true")
    parser.add_argument("--fixed_tile_attention", action="store_true")
    parser.add_argument("--batch_invariant_linear", action="store_true")
    args = parser.parse_args()
    global KV_SINK
    KV_SINK = args.kv_sink

    if args.mode in ("both", "encode"):
        if not args.input_file or not os.path.exists(args.input_file):
            raise FileNotFoundError(f"找不到输入文件: {args.input_file}")
    if args.mode == "decode" and not os.path.exists(args.compressed_file):
        raise FileNotFoundError(f"找不到压缩文件: {args.compressed_file}")

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

    print(f"Loading Model: {args.model_name} on {device} (dtype={args.dtype})")
    tokenizer = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=device,
        attn_implementation=args.attn_implementation,
    )
    if args.lora_path:
        from peft import PeftModel
        print(f"[Warn] 加载 LoRA {args.lora_path}：请确认它不是在测试文件上训练的")
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=dtype).merge_and_unload()
    model.eval()

    print(f"Loading Medusa heads from {args.medusa_heads_dir} (dtype={args.dtype})")
    medusa_heads = MedusaHeadCollection.load(args.medusa_heads_dir, device, dtype=dtype).eval()
    print(f"heads: K={medusa_heads.num_heads}, head_offset_start={medusa_heads.head_offset_start} "
          f"({'位置+1 用 backbone lm_head' if medusa_heads.uses_base_for_first else '旧约定：head0 预测 +1'})")

    if args.round_size is None:
        if args.num_active_heads is not None:
            args.round_size = args.num_active_heads + (1 if medusa_heads.uses_base_for_first else 0)
        else:
            args.round_size = medusa_heads.max_round_size
    if not (1 <= args.round_size <= medusa_heads.max_round_size):
        raise ValueError(f"round_size={args.round_size} 不合法，必须在 "
                         f"[1, {medusa_heads.max_round_size}] 范围内")
    print(f"round_size n = {args.round_size}: " + ", ".join(position_labels(medusa_heads, args.round_size)))

    linear_ctx = batch_invariant_linear_mode() if args.batch_invariant_linear else contextlib.nullcontext()
    with linear_ctx:
        ref_tokens = None
        if args.mode in ("both", "encode"):
            encoder = MedusaRankEncoder(model, medusa_heads, tokenizer, device, args)
            ref_tokens = encoder.encode_file(args.input_file, args.compressed_file)
            with open(args.compressed_file, 'rb') as f:
                print(f"压缩文件 sha256: {sha256_bytes(f.read())}")
        if args.mode in ("both", "decode"):
            decoder = MedusaRankDecoder(model, medusa_heads, tokenizer, device, args)
            decoded = decoder.decode_file(args.compressed_file, args.output_file)

    if args.mode in ("both", "decode"):
        if args.input_file and os.path.exists(args.input_file):
            ok = verify(tokenizer, args.input_file, args.output_file, decoded, ref_tokens)
            raise SystemExit(0 if ok else 1)
        print("（--input_file not exists，skip the lossnessless verify）")


if __name__ == "__main__":
    main()
