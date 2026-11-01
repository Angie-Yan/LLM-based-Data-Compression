
import torch
import triton
import triton.language as tl


@triton.jit
def _batch_invariant_causal_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Hq, Hkv, Sk, Dh,
    q_start_offset,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_q = tl.program_id(2).to(tl.int64)

    n_rep = Hq // Hkv
    kv_h = pid_h // n_rep

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < Dh

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + pid_q * stride_qm
    q = tl.load(q_ptr + d_offsets * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

    abs_q_pos = q_start_offset + pid_q  # 这一行 query 在拼接后序列里的绝对位置

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in range(0, Sk, BLOCK_N):
        n_offsets = (start_n + tl.arange(0, BLOCK_N)).to(tl.int64)
        kv_mask = n_offsets < Sk

        k_ptr = K + pid_b * stride_kb + kv_h * stride_kh + n_offsets[:, None] * stride_kn
        k = tl.load(
            k_ptr + d_offsets[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N]
        causal_mask = kv_mask & (n_offsets <= abs_q_pos)
        scores = tl.where(causal_mask, scores, float("-inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        # 整个 block 都被 mask 掉时 m_new 仍是 -inf，exp(-inf - (-inf)) 会是 nan，
        # 用 where 兜底成 0 贡献
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(causal_mask, tl.exp(scores - m_new), 0.0)

        v_ptr = V + pid_b * stride_vb + kv_h * stride_vh + n_offsets[:, None] * stride_vn
        v = tl.load(
            v_ptr + d_offsets[None, :] * stride_vd,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    acc = acc / l_i
    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_om
    tl.store(out_ptr + d_offsets * stride_od, acc, mask=d_mask)


def batch_invariant_causal_attention(query, key, value, scale, block_n=64):
    """
    query: [B, Hq, Sq, D]
    key/value: [B, Hkv, Sk, D]  (已经是"新 query 对应最后 Sq 个位置"的完整拼接序列)
    返回: [B, Hq, Sq, D]，float32
    """
    B, Hq, Sq, D = query.shape
    Hkv, Sk = key.shape[1], key.shape[2]
    assert Hq % Hkv == 0, "Hq 必须是 Hkv 的整数倍（GQA）"

    out = torch.empty_like(query, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    q_start_offset = Sk - Sq

    grid = (B, Hq, Sq)
    _batch_invariant_causal_attn_kernel[grid](
        query, key, value, out,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Hq, Hkv, Sk, D,
        q_start_offset, scale,
        BLOCK_N=block_n, BLOCK_D=BLOCK_D,
    )
    return out


def fp32_batch_invariant_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """跟 transformers 的 eager_attention_forward 签名兼容，可以直接注册进
    ALL_ATTENTION_FUNCTIONS["eager"]（参考 fast_rank_compresssion.py 里
    register_fp32_attention_patch 的做法）。"""
    def _repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)

    # 这里故意不调用 _repeat_kv：直接把原始（未展开）的 Hkv 传给 kernel，
    # 让 GQA 的头映射在 kernel 内部处理，省掉 repeat_kv 这一步矩阵拷贝。
    query_f32 = query.to(torch.float32)
    key_f32 = key.to(torch.float32)
    value_f32 = value.to(torch.float32)

    attn_output = batch_invariant_causal_attention(query_f32, key_f32, value_f32, scaling)
    attn_output = attn_output.to(query.dtype).transpose(1, 2).contiguous()
    return attn_output, None


# ==========================================
# 正确性验证：单条流、逐 token growing cache 的结果，
# 必须跟"一次性喂完整段"的结果逐位相同。
# 这就是我们真正需要保证的东西——先在这里过了再接入主流程。
# ==========================================
def test_batch_invariance():
    torch.manual_seed(0)
    device = "cuda"
    B, Hq, Hkv, D = 4, 8, 2, 64
    total_len = 40
    scale = D ** -0.5

    q_full = torch.randn(B, Hq, total_len, D, device=device)
    k_full = torch.randn(B, Hkv, total_len, D, device=device)
    v_full = torch.randn(B, Hkv, total_len, D, device=device)

    # 方式一：一次性把 40 个 query 全部喂进去（模拟 encode 的并行 chunk）
    out_parallel = batch_invariant_causal_attention(q_full, k_full, v_full, scale)

    # 方式二：逐个 token 喂（模拟 decode，每步 cache 长度 +1）
    out_sequential = torch.empty_like(out_parallel)
    for t in range(total_len):
        q_t = q_full[:, :, t:t + 1, :]
        k_t = k_full[:, :, :t + 1, :]
        v_t = v_full[:, :, :t + 1, :]
        out_t = batch_invariant_causal_attention(q_t, k_t, v_t, scale)
        out_sequential[:, :, t:t + 1, :] = out_t

    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


def test_batch_invariance():
    torch.manual_seed(0)
    device = "cuda"
    B, Hq, Hkv, D = 4, 8, 2, 64
    total_len = 40
    scale = D ** -0.5

    q_full = torch.randn(B, Hq, total_len, D, device=device)
    k_full = torch.randn(B, Hkv, total_len, D, device=device)
    v_full = torch.randn(B, Hkv, total_len, D, device=device)

    # 方式一：一次性把 40 个 query 全部喂进去（模拟 encode 的并行 chunk）
    out_parallel = batch_invariant_causal_attention(q_full, k_full, v_full, scale)

    # 方式二：逐个 token 喂（模拟 decode，每步 cache 长度 +1）
    out_sequential = torch.empty_like(out_parallel)
    for t in range(total_len):
        q_t = q_full[:, :, t:t + 1, :]
        k_t = k_full[:, :, :t + 1, :]
        v_t = v_full[:, :, :t + 1, :]
        out_t = batch_invariant_causal_attention(q_t, k_t, v_t, scale)
        out_sequential[:, :, t:t + 1, :] = out_t

    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


# ==========================================
# Batch-invariant Linear（自己写的，替代 batch_invariant_ops 里那个依赖新版
# Triton tl.range(flatten=True) 的 matmul_persistent 实现）。
#
# 覆盖的是 FFN（gate/up/down_proj）、QKVO 投影、lm_head 这些普通
# nn.Linear —— 跟 attention kernel 同样的"每个 program 只算 1 行输出"
# 原则：不管这次一起处理多少行（M），任意一行的计算路径完全一样。
# 只用最基础的 Triton 语法（tl.arange/tl.load/tl.store/tl.sum，没有用
# tl.range 的高级参数），兼容性应该比官方那个库好得多。
# ==========================================
@triton.jit
def _batch_invariant_linear_kernel(
    X, W, Bias, Out,
    N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)

    # 关键修复：pid_m * stride_om（行地址偏移）在 M、N 都比较大时会超过 int32
    # 上限（2^31-1），Triton 默认按 32 位算这个乘法会直接溢出、绕回成错误地址，
    # 表现为 "illegal memory access"。这里把参与地址计算的下标显式转成 int64，
    # 强制乘法按 64 位算，从根上避免溢出。
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_row_ptr = X + pid_m * stride_xm

    for k0 in range(0, K, BLOCK_K):
        k_idx = (k0 + offs_k).to(tl.int64)
        k_mask = k_idx < K
        x = tl.load(x_row_ptr + k_idx * stride_xk, mask=k_mask, other=0.0).to(tl.float32)
        w_ptr = W + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
        w = tl.load(w_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias

    out_ptr = Out + pid_m * stride_om + offs_n * stride_on
    tl.store(out_ptr, acc, mask=n_mask)


def batch_invariant_linear(x, weight, bias=None, block_k=128, block_n=128):
    """
    等价于 torch.nn.functional.linear(x, weight, bias)：
    x: [..., K]，weight: [N, K]，bias: [N] 或 None。
    返回 float32，shape [..., N]。
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    N = weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = (M, triton.cdiv(N, block_n))
    _batch_invariant_linear_kernel[grid](
        x2d, weight, bias if bias is not None else x2d, out,
        N, K,
        x2d.stride(0), x2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_K=block_k, BLOCK_N=block_n,
    )
    return out.reshape(*orig_shape[:-1], N)


# ==========================================
# 固定 tile 大小的 batch-invariant linear（同样思路：padding 到 BLOCK_M 的
# 整数倍，用 tl.dot 走 tensor core）。FFN(gate/up/down_proj)、QKVO 投影、
# lm_head 全部是普通 nn.Linear，计算量比 attention 大得多（每层 transformer
# block 有 7 次线性层调用，只有 1 次 attention），是目前速度的主要瓶颈。
# ==========================================
@triton.jit
def _fixed_tile_linear_kernel(
    X, W, Bias, Out,
    N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 调用方已保证 padding 过，不会越界
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = (k0 + offs_k).to(tl.int64)
        k_mask = k_idx < K

        x_ptr = X + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x = tl.load(x_ptr, mask=k_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_K]

        w_ptr = W + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
        w = tl.load(w_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x, tl.trans(w)).to(tl.float32)  # tensor core: [BLOCK_M, BLOCK_N]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    out_ptr = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr, acc, mask=n_mask[None, :])


def fixed_tile_batch_invariant_linear(x, weight, bias=None, block_m=64, block_n=64, block_k=64):
    orig_shape = x.shape
    K = orig_shape[-1]
    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    N = weight.shape[0]

    M_padded = ((M + block_m - 1) // block_m) * block_m
    if M_padded != M:
        pad = x2d.new_zeros(M_padded - M, K)
        x2d_padded = torch.cat([x2d, pad], dim=0)
    else:
        x2d_padded = x2d

    out_padded = torch.empty((M_padded, N), device=x.device, dtype=torch.float32)
    grid = (M_padded // block_m, triton.cdiv(N, block_n))
    _fixed_tile_linear_kernel[grid](
        x2d_padded, weight, bias if bias is not None else x2d_padded, out_padded,
        N, K,
        x2d_padded.stride(0), x2d_padded.stride(1),
        weight.stride(0), weight.stride(1),
        out_padded.stride(0), out_padded.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    out = out_padded[:M]
    return out.reshape(*orig_shape[:-1], N)


def test_fixed_tile_batch_invariant_linear():
    torch.manual_seed(0)
    device = "cuda"
    M, K, N = 300, 512, 1000
    dtype = torch.float16
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)

    out_full = fixed_tile_batch_invariant_linear(x, w, b)
    out_row0_alone = fixed_tile_batch_invariant_linear(x[:1], w, b)

    max_diff = (out_full[0] - out_row0_alone[0]).abs().max().item()
    print(f"[fixed_tile linear] 最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


class batch_invariant_linear_mode:
    """context manager：临时把 torch.nn.functional.linear 换成上面的
    batch-invariant 实现，同时把 Qwen2RMSNorm 也换成下面的 batch-invariant
    RMSNorm，退出时都还原。encode 和 decode 都要在同一个 context 里跑。"""
    def __enter__(self):
        import torch.nn.functional as F
        self._orig_linear = F.linear

        def _patched_linear(input, weight, bias=None):
            if input.is_cuda:
                return fixed_tile_batch_invariant_linear(input, weight, bias).to(input.dtype)
            return self._orig_linear(input, weight, bias)

        F.linear = _patched_linear

        self._rmsnorm_cls = None
        self._orig_rmsnorm_forward = None
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2_module
            self._rmsnorm_cls = qwen2_module.Qwen2RMSNorm
            self._orig_rmsnorm_forward = self._rmsnorm_cls.forward

            def _patched_rmsnorm_forward(rmsnorm_self, hidden_states):
                if not hidden_states.is_cuda:
                    return self._orig_rmsnorm_forward(rmsnorm_self, hidden_states)
                input_dtype = hidden_states.dtype
                eps = getattr(rmsnorm_self, "variance_epsilon", 1e-6)
                out = batch_invariant_rmsnorm(hidden_states, rmsnorm_self.weight.to(torch.float32), eps)
                return out.to(input_dtype)

            self._rmsnorm_cls.forward = _patched_rmsnorm_forward
        except Exception as e:
            print(f"[batch_invariant_linear_mode] RMSNorm 补丁未生效（{e!r}），"
                  f"F.linear 部分仍然会打补丁")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch.nn.functional as F
        F.linear = self._orig_linear
        if self._rmsnorm_cls is not None and self._orig_rmsnorm_forward is not None:
            self._rmsnorm_cls.forward = self._orig_rmsnorm_forward


def test_batch_invariant_linear():
    torch.manual_seed(0)
    device = "cuda"
    M, K, N = 300, 512, 1000
    x = torch.randn(M, K, device=device)
    w = torch.randn(N, K, device=device)
    b = torch.randn(N, device=device)

    out_full = batch_invariant_linear(x, w, b)         # 一次性算 300 行
    out_row0_alone = batch_invariant_linear(x[:1], w, b)  # 单独算第 1 行

    max_diff = (out_full[0] - out_row0_alone[0]).abs().max().item()
    print(f"最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


# ==========================================
# Batch-invariant RMSNorm —— thinking machines 那篇文章明确要求的第三块
# （RMSNorm / matmul / attention），之前只顾着搞定后两个，漏了这个。
# Qwen2RMSNorm 内部虽然自己会转 fp32，但 torch.mean() 这个归约本身的策略
# 也会随着一次处理多少行（M）变化，同样不是 batch invariant——这正是官方
# batch_invariant_ops 库要单独覆盖 torch.mean() 的原因。
# 同样是"每个 program 只算 1 行"：D（hidden_size）不随 M 变化，单独一行
# 的归约顺序天然固定，跟一起处理多少行完全无关。
# ==========================================
@triton.jit
def _batch_invariant_rmsnorm_kernel(
    X, W, Out,
    D, eps,
    stride_xm, stride_xd,
    stride_om, stride_od,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    x_ptr = X + pid_m * stride_xm
    x = tl.load(x_ptr + d_offsets * stride_xd, mask=d_mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    inv_rms = 1.0 / tl.sqrt(variance + eps)
    x_norm = x * inv_rms

    w = tl.load(W + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    out = x_norm * w

    out_ptr = Out + pid_m * stride_om
    tl.store(out_ptr + d_offsets * stride_od, out, mask=d_mask)


def batch_invariant_rmsnorm(x, weight, eps):
    orig_shape = x.shape
    D = orig_shape[-1]
    x2d = x.reshape(-1, D).contiguous()
    M = x2d.shape[0]
    out = torch.empty_like(x2d, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (M,)
    _batch_invariant_rmsnorm_kernel[grid](
        x2d, weight, out,
        D, eps,
        x2d.stride(0), x2d.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return out.reshape(orig_shape)


def test_batch_invariant_rmsnorm():
    torch.manual_seed(0)
    device = "cuda"
    M, D = 300, 896
    x = torch.randn(M, D, device=device)
    w = torch.randn(D, device=device)
    eps = 1e-6

    out_full = batch_invariant_rmsnorm(x, w, eps)
    out_row0_alone = batch_invariant_rmsnorm(x[:1], w, eps)

    max_diff = (out_full[0] - out_row0_alone[0]).abs().max().item()
    print(f"最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


# ==========================================
# 固定 tile 大小的 causal attention（用 tensor core，速度和正确性两不误）
#
# 跟前面 batch_invariant_causal_attention 的区别：不再是"每个 program 只算
# 1 行"，而是"每个 program 永远处理 BLOCK_M 行"——不管这次真实要算多少个
# query。真实行数不够 BLOCK_M 的整数倍时（decode 的 Sq=1 是最极端的情况），
# 在调用方把 Q 填充（padding）到 BLOCK_M 的整数倍，填充的是什么值完全不重要：
# attention 里不同 query 之间没有交互——每一行的输出只取决于它自己的 Q 和
# 全部 K/V，不受同一批里其他行装的是真数据还是垃圾数据影响。需要保证的只是
# "这次调用用的 tile 大小固定不变"，这样 kernel 内部选择的计算路径
# （包括是否用 tensor core、reduction 顺序）就不会因为真实 query 数量变化
# 而改变。算完之后只取真实的前 Sq 行，扔掉填充的部分。
#
# QK^T、P@V 都用 tl.dot（会调用 tensor core），Q/K/V 保持原始 dtype
# （比如 fp16）传入 —— tensor core 原生支持 fp16 输入配 fp32 累加器，
# 这是标准 FlashAttention 的做法，比之前那版全程手动转 fp32 再逐元素乘
# 快得多，也一样能利用 tensor core 内部固定的 fp32 累加精度。
# ==========================================
@triton.jit
def _fixed_tile_causal_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Hq, Hkv, Sk, Dh,
    q_start_offset,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_m_block = tl.program_id(2).to(tl.int64)

    n_rep = Hq // Hkv
    kv_h = pid_h // n_rep

    m_offsets = pid_m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # 调用方已保证这些下标不越界（padding 过）
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < Dh

    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh \
        + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptr, mask=d_mask[None, :], other=0.0)  # [BLOCK_M, BLOCK_D]，保持原 dtype

    abs_q_pos = q_start_offset + m_offsets  # [BLOCK_M]；填充行算出来的绝对位置没意义，反正会被丢弃

    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, Sk, BLOCK_N):
        n_offsets = (start_n + tl.arange(0, BLOCK_N)).to(tl.int64)
        kv_mask = n_offsets < Sk

        k_ptr = K + pid_b * stride_kb + kv_h * stride_kh \
            + n_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptr, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_D]

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale  # [BLOCK_M, BLOCK_N]，tensor core

        causal_mask = kv_mask[None, :] & (n_offsets[None, :] <= abs_q_pos[:, None])
        scores = tl.where(causal_mask, scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(causal_mask, tl.exp(scores - m_new[:, None]), 0.0)

        v_ptr = V + pid_b * stride_vb + kv_h * stride_vh \
            + n_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptr, mask=kv_mask[:, None] & d_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_D]

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptr = Out + pid_b * stride_ob + pid_h * stride_oh \
        + m_offsets[:, None] * stride_om + d_offsets[None, :] * stride_od
    tl.store(out_ptr, acc, mask=d_mask[None, :])


def fixed_tile_batch_invariant_attention(query, key, value, scale, block_m=64, block_n=64):
    """
    query: [B, Hq, Sq, D]
    key/value: [B, Hkv, Sk, D]（新 query 对应最后 Sq 个位置）
    返回: [B, Hq, Sq, D]，float32
    """
    B, Hq, Sq, D = query.shape
    Hkv, Sk = key.shape[1], key.shape[2]
    assert Hq % Hkv == 0

    Sq_padded = ((Sq + block_m - 1) // block_m) * block_m
    if Sq_padded != Sq:
        pad = query.new_zeros(B, Hq, Sq_padded - Sq, D)
        query_padded = torch.cat([query, pad], dim=2)
    else:
        query_padded = query

    out_padded = torch.empty(B, Hq, Sq_padded, D, device=query.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    q_start_offset = Sk - Sq

    grid = (B, Hq, Sq_padded // block_m)
    _fixed_tile_causal_attn_kernel[grid](
        query_padded, key, value, out_padded,
        query_padded.stride(0), query_padded.stride(1), query_padded.stride(2), query_padded.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        out_padded.stride(0), out_padded.stride(1), out_padded.stride(2), out_padded.stride(3),
        Hq, Hkv, Sk, D,
        q_start_offset, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
    )
    return out_padded[:, :, :Sq, :]


def fixed_tile_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """跟 transformers 的 eager_attention_forward 签名兼容。"""
    attn_output = fixed_tile_batch_invariant_attention(query, key, value, scaling)
    attn_output = attn_output.to(query.dtype).transpose(1, 2).contiguous()
    return attn_output, None


def test_fixed_tile_batch_invariance(block_m=64):
    torch.manual_seed(0)
    device = "cuda"
    B, Hq, Hkv, D = 4, 8, 2, 64
    total_len = 40
    scale = D ** -0.5
    dtype = torch.float16

    q_full = torch.randn(B, Hq, total_len, D, device=device, dtype=dtype)
    k_full = torch.randn(B, Hkv, total_len, D, device=device, dtype=dtype)
    v_full = torch.randn(B, Hkv, total_len, D, device=device, dtype=dtype)

    # 方式一：一次性把 40 个 query 全部喂进去（会被 padding 到 block_m 的整数倍）
    out_parallel = fixed_tile_batch_invariant_attention(q_full, k_full, v_full, scale, block_m=block_m)

    # 方式二：逐个 token 喂（每次都会被单独 padding 到 block_m）
    out_sequential = torch.empty_like(out_parallel)
    for t in range(total_len):
        q_t = q_full[:, :, t:t + 1, :]
        k_t = k_full[:, :, :t + 1, :]
        v_t = v_full[:, :, :t + 1, :]
        out_t = fixed_tile_batch_invariant_attention(q_t, k_t, v_t, scale, block_m=block_m)
        out_sequential[:, :, t:t + 1, :] = out_t

    max_diff = (out_parallel - out_sequential).abs().max().item()
    print(f"[fixed_tile, block_m={block_m}] 最大误差: {max_diff}")
    print("PASS（逐位相同）" if max_diff == 0.0 else "FAIL（还不是 batch invariant）")
    return max_diff == 0.0


if __name__ == "__main__":
    test_batch_invariance()
    test_batch_invariant_linear()
    test_batch_invariant_rmsnorm()
    test_fixed_tile_batch_invariance()
    test_fixed_tile_batch_invariant_linear()
