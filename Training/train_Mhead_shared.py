"""
Medusa Head Training（domain-level）
=========================================

用法：
  # Wiki：training set = file under path - 10 test files；keep 5 files ramdomly to verify
  python train_Mhead_shared.py train \
      --model_name Qwen/Qwen2.5-0.5B \
      --train_files "/data/wiki/*.txt" \
      --test_files  "/data/wiki_test/*.txt" \
      --val_num_files 5 \
      --output_dir ./medusa_domain_wiki \
      --num_heads 5 --max_length 1024 --batch_size 4 --bf16

  # evaluation on 10 test files
  python train_Mhead_shared.py eval \
      --model_name Qwen/Qwen2.5-0.5B \
      --medusa_heads_dir ./medusa_domain_wiki/best \
      --test_files "/data/wiki_test/*.txt" \
      --context_length 1024
"""

import os
import glob
import json
import time
import random
import hashlib
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


# ==========================================
# Medusa Head 结构（与原脚本相同）
# ==========================================

class MedusaHead(nn.Module):
    """Linear -> SiLU -> residual -> lm_head；linear 零初始化，lm_head 从 base 拷贝。"""
    def __init__(self, hidden_size, vocab_size, init_lm_head_weight=None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.linear.weight)
        if init_lm_head_weight is not None:
            with torch.no_grad():
                self.lm_head.weight.copy_(init_lm_head_weight)

    def forward(self, hidden_states):
        x = self.act(self.linear(hidden_states)) + hidden_states
        return self.lm_head(x)


class MedusaHeadCollection(nn.Module):
    """
    K 个 head。第 k 个 head（0-indexed）预测当前位置之后第 head_offset_start + k 个 token。
    head_offset_start=2：原版 Medusa 约定（+1 由 base lm_head 负责）。
    head_offset_start=1：旧脚本约定（head0 重复预测 +1）。
    """
    def __init__(self, hidden_size, vocab_size, num_heads,
                 init_lm_head_weight=None, head_offset_start=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_offset_start = head_offset_start
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, init_lm_head_weight)
            for _ in range(num_heads)
        ])

    def offset(self, k):
        return self.head_offset_start + k

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]

    def save(self, save_dir, state_dict=None):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict if state_dict is not None else self.state_dict(),
                   os.path.join(save_dir, "medusa_heads.pt"))
        config = {
            "num_heads": self.num_heads,
            "hidden_size": self.heads[0].linear.in_features,
            "vocab_size": self.heads[0].lm_head.out_features,
            "head_offset_start": self.head_offset_start,
        }
        with open(os.path.join(save_dir, "medusa_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, save_dir, device="cuda"):
        with open(os.path.join(save_dir, "medusa_config.json")) as f:
            config = json.load(f)
        # 旧 checkpoint 没有这个字段 => 旧约定 offset 从 1 开始
        heads = cls(config["hidden_size"], config["vocab_size"], config["num_heads"],
                    head_offset_start=config.get("head_offset_start", 1))
        state = torch.load(os.path.join(save_dir, "medusa_heads.pt"), map_location=device)
        heads.load_state_dict(state)
        return heads.to(device)


# ==========================================
# 文件收集 / 划分 / 泄漏检查
# ==========================================

def expand_files(specs):
    """支持：具体路径、glob、目录（取目录下所有文件）、@list.txt（每行一个路径）。"""
    out = []
    for s in specs or []:
        if s.startswith("@"):
            with open(s[1:]) as f:
                out += [l.strip() for l in f if l.strip()]
        elif os.path.isdir(s):
            out += sorted(p for p in glob.glob(os.path.join(s, "*")) if os.path.isfile(p))
        else:
            out += sorted(glob.glob(s))
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def norm_lines(path, min_len):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = " ".join(line.split())
            if len(s) >= min_len:
                yield s


def split_and_check(train_specs, val_specs, test_specs, val_num_files, seed,
                    min_line_len, max_overlap):
    train_files = expand_files(train_specs)
    test_files = expand_files(test_specs)
    test_real = {os.path.realpath(p) for p in test_files}
    test_sha = {file_sha(p) for p in test_files}

    before = len(train_files)
    train_files = [p for p in train_files
                   if os.path.realpath(p) not in test_real and file_sha(p) not in test_sha]
    print(f"[Split] train candidates {before} -> {len(train_files)} after removing test files")

    if val_specs:
        val_files = expand_files(val_specs)
        bad = [p for p in val_files if os.path.realpath(p) in test_real or file_sha(p) in test_sha]
        assert not bad, f"val files overlap test files: {bad}"
        val_set = {os.path.realpath(p) for p in val_files}
        train_files = [p for p in train_files if os.path.realpath(p) not in val_set]
    else:
        rng = random.Random(seed)
        shuffled = train_files[:]
        rng.shuffle(shuffled)
        val_files = sorted(shuffled[:val_num_files])
        train_files = sorted(shuffled[val_num_files:])
    assert train_files and val_files, "train/val 为空，检查路径"
    print(f"[Split] train={len(train_files)} files, val={len(val_files)} files, "
          f"test={len(test_files)} files")

    # 行级重叠：测试文件的“长行”有多大比例在训练集里出现过
    if test_files:
        train_hashes = set()
        for p in train_files + val_files:
            for s in norm_lines(p, min_line_len):
                train_hashes.add(hashlib.md5(s.encode()).digest())
        worst = 0.0
        for p in test_files:
            lines = list(norm_lines(p, min_line_len))
            if not lines:
                continue
            hit = sum(hashlib.md5(s.encode()).digest() in train_hashes for s in lines)
            frac = hit / len(lines)
            worst = max(worst, frac)
            print(f"[Leak] {os.path.basename(p)}: {hit}/{len(lines)} long lines "
                  f"also in train/val ({frac:.1%})")
        if worst > max_overlap:
            raise RuntimeError(
                f"测试文件与训练集行级重叠最高达 {worst:.1%} > {max_overlap:.1%}。"
                f"Code 语料请先按来源（MBPP task id / RocksDB 文件）去重划分；"
                f"确认无误可调大 --max_overlap。")

    return train_files, val_files, test_files


# ==========================================
# 数据集
# ==========================================

class TokenChunkDataset(Dataset):
    """逐文件 tokenize 后拼接（文件间插入 eos），按 max_length 切成不重叠的块。"""
    def __init__(self, files, tokenizer, max_length, max_tokens=-1):
        sep = tokenizer.eos_token_id
        ids = []
        for p in files:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                ids += tokenizer.encode(f.read(), add_special_tokens=False)
            if sep is not None:
                ids.append(sep)
            if 0 < max_tokens <= len(ids):
                ids = ids[:max_tokens]
                break
        t = torch.tensor(ids, dtype=torch.long)
        n = (len(t) - 1) // max_length
        self.chunks = t[: n * max_length].view(n, max_length)
        print(f"[Data] {len(files)} files -> {len(t):,} tokens -> {n} chunks of {max_length}")

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ==========================================
# Base model
# ==========================================

def load_base(model_name, lora_path, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                 device_map=device).eval()
    if lora_path:
        # 只能是 domain-level 的 LoRA；per-file LoRA 本身就是要计入压缩大小的数据
        from peft import PeftModel
        print(f"[Warn] loading LoRA {lora_path}: make sure it is NOT trained on test files")
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=dtype).merge_and_unload()
    for p in model.parameters():
        p.requires_grad = False
    return model


def normed_hidden(base_model, input_ids):
    """取最后一层 hidden 并过 final norm（与 base lm_head 的输入一致）。"""
    out = base_model.model(input_ids=input_ids, use_cache=False)
    return out.last_hidden_state  # Qwen2Model 的 last_hidden_state 已经过 final norm


# ==========================================
# 训练
# ==========================================

@torch.no_grad()
def val_losses(base_model, heads, loader, device, max_batches, autocast_ctx):
    heads.eval()
    sums = [0.0] * heads.num_heads
    accs = [0.0] * heads.num_heads
    n = 0
    for b, input_ids in enumerate(loader):
        if b >= max_batches:
            break
        input_ids = input_ids.to(device)
        with autocast_ctx():
            hidden = normed_hidden(base_model, input_ids)
            for k, head in enumerate(heads.heads):
                o = heads.offset(k)
                logits = head(hidden[:, :-o]).float()
                tgt = input_ids[:, o:]
                sums[k] += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           tgt.reshape(-1)).item()
                accs[k] += (logits.argmax(-1) == tgt).float().mean().item()
                del logits
        n += 1
    heads.train()
    return [s / max(n, 1) for s in sums], [a / max(n, 1) for a in accs]


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True  # 只影响训练，和编解码无损无关
    torch.backends.cudnn.allow_tf32 = True

    train_files, val_files, _ = split_and_check(
        args.train_files, args.val_files, args.test_files, args.val_num_files,
        args.seed, args.min_line_len, args.max_overlap)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"train": train_files, "val": val_files,
                   "test": expand_files(args.test_files)}, f, indent=2)

    base_model = load_base(args.model_name, args.lora_path, device, torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.lm_head.weight.shape[0]

    heads = MedusaHeadCollection(hidden_size, vocab_size, args.num_heads,
                                 init_lm_head_weight=base_model.lm_head.weight.data.clone(),
                                 head_offset_start=2).to(device)
    if args.resume_from:
        heads.load_state_dict(torch.load(os.path.join(args.resume_from, "medusa_heads.pt"),
                                         map_location=device))

    res_params, lm_params = [], []
    for h in heads.heads:
        res_params += list(h.linear.parameters())
        if args.freeze_head_lm:
            h.lm_head.weight.requires_grad = False
        else:
            lm_params += list(h.lm_head.parameters())
    groups = [{"params": res_params, "lr": args.lr}]
    if lm_params:
        groups.append({"params": lm_params,
                       "lr": args.lm_head_lr if args.lm_head_lr > 0 else args.lr / 10})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"[Model] heads={args.num_heads} (offsets +2..+{args.num_heads + 1}), "
          f"trainable params={n_train:,}, freeze_head_lm={args.freeze_head_lm}")

    train_ds = TokenChunkDataset(train_files, tokenizer, args.max_length, args.max_tokens)
    val_ds = TokenChunkDataset(val_files, tokenizer, args.max_length, -1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    # 固定的随机子集：每次验证都用同一批 chunk，保证各次 val loss 可比
    n_val = min(len(val_ds), args.val_batches * args.batch_size)
    val_idx = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(0))[:n_val]
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx.tolist()),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)

    if args.bf16:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext

    # 每个 head 独立记录 best（backbone 冻结 => 各 head 的 loss 互不影响）
    best_loss = [float("inf")] * args.num_heads
    best_state = [None] * args.num_heads
    bad_evals = 0
    step = 0
    t0 = time.perf_counter()

    def do_eval():
        nonlocal bad_evals
        vl, va = val_losses(base_model, heads, val_loader, device, args.val_batches, autocast_ctx)
        improved = False
        for k in range(args.num_heads):
            if vl[k] < best_loss[k] - 1e-4:
                best_loss[k] = vl[k]
                best_state[k] = {n: p.detach().to("cpu", copy=True)
                                 for n, p in heads.heads[k].state_dict().items()}
                improved = True
        bad_evals = 0 if improved else bad_evals + 1
        bits = " ".join(f"+{heads.offset(k)}:{vl[k] / np.log(2):.2f}b/{va[k]:.3f}"
                        for k in range(args.num_heads))
        print(f"[Val] step {step} | bits/tok & top1 per offset: {bits} | "
              f"improved={improved} bad_evals={bad_evals}", flush=True)
        if improved:
            save_best()

    def save_best():
        sd = {}
        for k in range(args.num_heads):
            src = best_state[k] if best_state[k] is not None else heads.heads[k].state_dict()
            for n, p in src.items():
                sd[f"heads.{k}.{n}"] = p
        heads.save(os.path.join(args.output_dir, "best"), state_dict=sd)
        with open(os.path.join(args.output_dir, "best", "val_bits.json"), "w") as f:
            json.dump({f"+{heads.offset(k)}": best_loss[k] / np.log(2)
                       for k in range(args.num_heads)}, f, indent=2)

    do_eval()  # step 0：零初始化 head 的基线（等价于“直接用 base lm_head 预测更远位置”）
    stop = False
    for epoch in range(args.num_epochs):
        for input_ids in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.no_grad(), autocast_ctx():
                hidden = normed_hidden(base_model, input_ids).detach()

            optimizer.zero_grad(set_to_none=True)
            losses = []
            for k, head in enumerate(heads.heads):
                o = heads.offset(k)
                with autocast_ctx():
                    logits = head(hidden[:, :-o])
                loss_k = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                         input_ids[:, o:].reshape(-1))
                (loss_k / args.num_heads).backward()   # 逐 head 反向，省显存
                losses.append(loss_k.item())
                del logits, loss_k
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                s = " ".join(f"+{heads.offset(k)}:{l / np.log(2):.2f}b"
                             for k, l in enumerate(losses))
                print(f"[Train] ep {epoch + 1} step {step}/{total_steps} | {s} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            if step % args.eval_every == 0:
                do_eval()
                if bad_evals >= args.patience:
                    print(f"[EarlyStop] no head improved for {args.patience} evals")
                    stop = True
                    break
        if stop:
            break

    if step % args.eval_every != 0:
        do_eval()
    print(f"\nDone. best val bits/token per offset: "
          + " ".join(f"+{heads.offset(k)}:{best_loss[k] / np.log(2):.3f}"
                     for k in range(args.num_heads)))
    print(f"Checkpoint: {os.path.join(args.output_dir, 'best')}")


# ==========================================
# Evaluation
# ==========================================

@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = load_base(args.model_name, args.lora_path, device, torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    heads = MedusaHeadCollection.load(args.medusa_heads_dir, device).eval()
    if heads.head_offset_start != 2:
        print(f"[Warn] checkpoint head_offset_start={heads.head_offset_start} (旧约定)")

    # predictors: (名字, offset, 取 logits 的函数)
    preds = [("base", 1, lambda h: base_model.lm_head(h))]
    for k, head in enumerate(heads.heads):
        preds.append((f"head{k}", heads.offset(k), head))
    max_off = max(o for _, o, _ in preds)

    L = args.context_length
    per_file = []
    for path in expand_files(args.test_files):
        with open(path, "rb") as f:
            raw = f.read()
        ids = tokenizer.encode(raw.decode("utf-8", errors="ignore"), add_special_tokens=False)
        t = torch.tensor(ids, dtype=torch.long)
        n_win = len(t) // L
        windows = t[: n_win * L].view(n_win, L)
        # 每个 offset：bits 总和、计数、top1、rank 分桶
        stats = {name: dict(bits=0.0, cnt=0, top1=0, r2_4=0, r5_32=0, r33=0)
                 for name, _, _ in preds}
        for s in range(0, n_win, args.eval_batch):
            x = windows[s: s + args.eval_batch].to(device)
            hidden = normed_hidden(base_model, x)
            # 只在所有 offset 都有标签的位置上统计，保证各 offset 的样本集合一致
            hv = hidden[:, : L - max_off]
            for name, o, fn in preds:
                logits = fn(hv).float()
                tgt = x[:, o: o + hv.size(1)]
                lp = torch.log_softmax(logits, -1)
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                tgt_logit = logits.gather(-1, tgt.unsqueeze(-1))
                rank = (logits > tgt_logit).sum(-1) + 1
                st = stats[name]
                st["bits"] += (nll / np.log(2)).sum().item()
                st["cnt"] += nll.numel()
                st["top1"] += (rank == 1).sum().item()
                st["r2_4"] += ((rank >= 2) & (rank <= 4)).sum().item()
                st["r5_32"] += ((rank >= 5) & (rank <= 32)).sum().item()
                st["r33"] += (rank > 32).sum().item()
                del logits, lp
        n_tok = n_win * L
        bytes_per_tok = len(raw) / max(len(ids), 1)
        bpt = [stats[name]["bits"] / stats[name]["cnt"] for name, _, _ in preds]
        # 一轮 n 个位置：位置 1 用 base，位置 j 用 offset=j 的预测器；每个预测器负责 1/n 的 token
        ideal_bpb = {n: float(np.mean(bpt[:n]) / bytes_per_tok) for n in range(1, len(preds) + 1)}
        per_file.append(dict(file=os.path.basename(path), bytes_per_tok=bytes_per_tok,
                             bits_per_tok=bpt, ideal_bpb=ideal_bpb, stats=stats))
        print(f"\n{os.path.basename(path)}  ({len(raw):,} B, {len(ids):,} tok, "
              f"{bytes_per_tok:.2f} B/tok, windows of {L})")
        for (name, o, _), b in zip(preds, bpt):
            st = stats[name]
            c = st["cnt"]
            print(f"  {name:6s} +{o}: {b:6.3f} bits/tok | top1 {st['top1'] / c:.3f} | "
                  f"r2-4 {st['r2_4'] / c:.3f} | r5-32 {st['r5_32'] / c:.3f} | "
                  f">32 {st['r33'] / c:.3f}")
        print("  ideal bpb by n: " + " ".join(f"n={n}:{v:.3f}" for n, v in ideal_bpb.items()))

    if per_file:
        print("\n==== mean ± std over files ====")
        names = [f"{name}(+{o})" for name, o, _ in preds]
        arr = np.array([f["bits_per_tok"] for f in per_file])
        for i, nm in enumerate(names):
            print(f"  {nm:10s} bits/tok {arr[:, i].mean():.3f} ± {arr[:, i].std():.3f}")
        for n in range(1, len(preds) + 1):
            v = np.array([f["ideal_bpb"][n] for f in per_file])
            print(f"  ideal bpb n={n}: {v.mean():.3f} ± {v.std():.3f}")
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(per_file, f, indent=2)


# ==========================================
# Main
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")

    tp = sub.add_parser("train")
    tp.add_argument("--model_name", required=True)
    tp.add_argument("--lora_path", default="", help="Not used in experiment. Only allows domain-level LoRA")
    tp.add_argument("--train_files", nargs="+", required=True,
                    help="路径/glob/目录/@list.txt；会自动剔除 test 文件")
    tp.add_argument("--val_files", nargs="*", default=None,
                    help="不给则从 train 中随机留 --val_num_files 个文件")
    tp.add_argument("--val_num_files", type=int, default=5)
    tp.add_argument("--test_files", nargs="*", default=None, help="用于排除和泄漏检查")
    tp.add_argument("--min_line_len", type=int, default=40, help="泄漏检查只看长度>=此值的行")
    tp.add_argument("--max_overlap", type=float, default=0.05,
                    help="任一测试文件的长行在训练集中出现比例超过此值则报错")
    tp.add_argument("--output_dir", default="./medusa_domain")
    tp.add_argument("--num_heads", type=int, default=5, help="K 个 head => n 最大 K+1")
    tp.add_argument("--num_epochs", type=int, default=2)
    tp.add_argument("--batch_size", type=int, default=4)
    tp.add_argument("--max_length", type=int, default=1024, help="建议与推理 context 一致")
    tp.add_argument("--max_tokens", type=int, default=-1, help="-1 = 全部")
    tp.add_argument("--lr", type=float, default=1e-3, help="ResBlock 学习率")
    tp.add_argument("--lm_head_lr", type=float, default=-1, help="默认 lr/10")
    tp.add_argument("--freeze_head_lm", action="store_true",
                    help="冻结各 head 的 lm_head 副本，只训练 ResBlock")
    tp.add_argument("--weight_decay", type=float, default=0.01)
    tp.add_argument("--bf16", action="store_true", help="训练时 bf16 autocast（推荐）")
    tp.add_argument("--log_every", type=int, default=100)
    tp.add_argument("--eval_every", type=int, default=1000)
    tp.add_argument("--val_batches", type=int, default=100)
    tp.add_argument("--patience", type=int, default=3)
    tp.add_argument("--resume_from", default="")
    tp.add_argument("--seed", type=int, default=0)

    ep = sub.add_parser("eval")
    ep.add_argument("--model_name", required=True)
    ep.add_argument("--lora_path", default="")
    ep.add_argument("--medusa_heads_dir", required=True)
    ep.add_argument("--test_files", nargs="+", required=True)
    ep.add_argument("--context_length", type=int, default=1024)
    ep.add_argument("--eval_batch", type=int, default=2)
    ep.add_argument("--out_json", default="")

    args = ap.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
