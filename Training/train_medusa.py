"""
Medusa Head 训练脚本 - 针对 Qwen 压缩场景
==========================================
训练策略：Medusa-1，冻结 base model，只训练 heads
目标：提升每个 head 的 top-1 命中率，加速 encode 端的 rank 编码

用法：
  # 训练 0.5B 的 Medusa heads
  python train_medusa.py \
      --model_name Qwen/Qwen1.5-0.5B \
      --lora_path /path/to/lora \   # 可选，有finetune权重就加
      --data_file /path/to/wiki.txt \
      --output_dir ./medusa_heads_0.5b \
      --num_heads 4 \
      --num_epochs 3

  # 训练 7B 的 Medusa heads
  python train_medusa.py \
      --model_name Qwen/Qwen1.5-7B \
      --data_file /path/to/wiki.txt \
      --output_dir ./medusa_heads_7b \
      --num_heads 4 \
      --num_epochs 3
"""

import os
import math
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import PeftModel


# ==========================================
# Medusa Head 结构
# ==========================================

class MedusaHead(nn.Module):
    """
    单个 Medusa head：一层 FFN + 残差连接 + LM head
    结构和原论文一致：Linear -> SiLU -> residual -> lm_head

    关键初始化策略（对齐官方 Medusa 实现）：
    - linear 层零初始化：训练开始时 SiLU(0 @ hidden) = 0，
      残差分支退化为恒等映射，head 的初始输出 = lm_head(hidden)
    - lm_head 复用 base_model 训练好的 lm_head 权重，而不是随机初始化

    这样 head 在训练第一步就已经具备和 base_model 同等的预测能力，
    之后的训练只是在这个基础上学习"如何用当前位置的 hidden 预测
    更远的未来 token"，而不是从零学一个全新的 1亿+ 参数分类头。
    不做这个初始化会导致收敛极慢，训练初期 acc 长期接近随机水平。
    """
    def __init__(self, hidden_size: int, vocab_size: int,
                 init_lm_head_weight: torch.Tensor = None):
        super().__init__()
        self.linear  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act     = nn.SiLU()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 零初始化 linear：让初始残差等价于恒等映射
        nn.init.zeros_(self.linear.weight)

        # 复用 base_model 的 lm_head 权重作为初始值
        if init_lm_head_weight is not None:
            with torch.no_grad():
                self.lm_head.weight.copy_(init_lm_head_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, seq_len, hidden_size]
        返回: [B, seq_len, vocab_size]
        """
        x = self.act(self.linear(hidden_states))
        x = x + hidden_states          # 残差连接
        return self.lm_head(x)


class MedusaHeadCollection(nn.Module):
    """
    K 个 Medusa heads 的集合，挂在 base model 上
    第 k 个 head 预测当前位置后第 k+1 个 token
    """
    def __init__(self, hidden_size: int, vocab_size: int, num_heads: int,
                 init_lm_head_weight: torch.Tensor = None):
        super().__init__()
        self.num_heads = num_heads
        # 每个 head 各自拷贝一份 lm_head 初始权重（深拷贝，训练时各自独立更新）
        self.heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, init_lm_head_weight)
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states: torch.Tensor):
        """
        返回: list of [B, seq_len, vocab_size]，长度为 num_heads
        """
        return [head(hidden_states) for head in self.heads]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "medusa_heads.pt"))
        config = {
            "num_heads":   self.num_heads,
            "hidden_size": self.heads[0].linear.in_features,
            "vocab_size":  self.heads[0].lm_head.out_features,
        }
        with open(os.path.join(save_dir, "medusa_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Save] Medusa heads saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, device: str = "cuda"):
        with open(os.path.join(save_dir, "medusa_config.json")) as f:
            config = json.load(f)
        heads = cls(config["hidden_size"], config["vocab_size"], config["num_heads"])
        state = torch.load(
            os.path.join(save_dir, "medusa_heads.pt"),
            map_location=device
        )
        heads.load_state_dict(state)
        return heads.to(device)


# ==========================================
# 数据集
# ==========================================

class TextTokenDataset(Dataset):
    """
    把文本文件 tokenize 后，按 max_length 切块。
    每个样本是一段连续的 token ids。
    """
    def __init__(self, file_path: str, tokenizer, max_length: int = 512,
                 max_tokens: int = 2_000_000):
        print(f"Loading and tokenizing {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        all_ids = tokenizer.encode(text, add_special_tokens=True)
        all_ids = all_ids[:max_tokens]
        print(f"Total tokens: {len(all_ids)}")

        # 切块，丢掉不足一块的尾巴
        self.chunks = []
        for start in range(0, len(all_ids) - max_length, max_length):
            chunk = all_ids[start: start + max_length]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Total chunks: {len(self.chunks)} (length={max_length})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


# ==========================================
# 训练逻辑
# ==========================================

def get_normed_hidden(base_model, outputs):
    """
    取最后一层 hidden states，并应用最终的 norm 层，
    使其分布和原始 lm_head 的输入一致。

    原因：outputs.hidden_states[-1] 是最后一个 decoder layer 的
    原始输出（未norm），而 outputs.logits 实际是
    lm_head(norm(hidden_states[-1])) 算出来的。
    Medusa head 必须吃 norm 之后的 hidden，否则数值分布和
    base_model 自己的 lm_head 训练时见过的完全不一样，会导致
    head 难以收敛。
    """
    hidden = outputs.hidden_states[-1]

    # 找到 norm 层，兼容 PeftModel 包装和普通 model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'norm'):
        norm_layer = base_model.model.norm
    elif hasattr(base_model, 'base_model'):
        # PeftModel 情况（理论上 merge_and_unload 后不会走这支）
        norm_layer = base_model.base_model.model.norm
    else:
        raise AttributeError("找不到 norm 层，请检查模型结构")

    return norm_layer(hidden)


def compute_loss_and_acc(base_hidden, medusa_heads, input_ids, num_heads):
    """
    计算所有 head 的 CE loss 和 top-1 accuracy。

    base_hidden: [B, seq_len, hidden]  base model 最后一层 hidden states
    input_ids:   [B, seq_len]          原始 token ids（label）

    第 k 个 head 在位置 t 预测 input_ids[:, t+k+1]
    所以第 k 个 head 的有效预测范围是 [:, :seq_len-k-1] → [:, k+1:]
    """
    all_logits = medusa_heads(base_hidden)  # list of [B, seq, V]，长度 num_heads

    total_loss = 0.0
    accs = []

    for k, logits_k in enumerate(all_logits):
        offset = k + 1                   # 第 k 个 head 预测 offset 步之后的 token
        # 去掉最后 offset 个位置（没有对应的 label）
        pred   = logits_k[:, :-offset, :].contiguous()   # [B, seq-offset, V]
        target = input_ids[:, offset:].contiguous()       # [B, seq-offset]

        B, L, V = pred.shape
        loss_k = F.cross_entropy(
            pred.view(B * L, V),
            target.view(B * L)
        )
        total_loss += loss_k

        # top-1 accuracy
        with torch.no_grad():
            top1 = pred.argmax(dim=-1)    # [B, L]
            acc_k = (top1 == target).float().mean().item()
            accs.append(acc_k)

    return total_loss / num_heads, accs


@torch.no_grad()
def evaluate(base_model, medusa_heads, dataloader, device, num_heads, max_batches=50):
    """快速评估：在 max_batches 个 batch 上跑 val loss 和各 head accuracy"""
    base_model.eval()
    medusa_heads.eval()

    total_loss = 0.0
    total_accs = [0.0] * num_heads
    count = 0

    for batch_idx, input_ids in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device)

        outputs = base_model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True
        )
        hidden = get_normed_hidden(base_model, outputs)  # [B, seq, hidden]，已norm

        loss, accs = compute_loss_and_acc(hidden, medusa_heads, input_ids, num_heads)
        total_loss += loss.item()
        for k in range(num_heads):
            total_accs[k] += accs[k]
        count += 1

    avg_loss = total_loss / max(count, 1)
    avg_accs = [a / max(count, 1) for a in total_accs]
    return avg_loss, avg_accs


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- 加载 base model（冻结）----
    print(f"\nLoading base model: {args.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=device
    ).eval()

    # 加载 LoRA（如果有）
    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path} ...")
        base_model = PeftModel.from_pretrained(
            base_model, args.lora_path, torch_dtype=torch.float32
        )
        base_model = base_model.merge_and_unload()
        print("LoRA merged.")

    # 冻结 base model 所有参数
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    print(f"Base model frozen. Params: {sum(p.numel() for p in base_model.parameters()):,}")

    # ---- 初始化 Medusa heads ----
    hidden_size = base_model.config.hidden_size
    vocab_size  = base_model.config.vocab_size
    print(f"hidden_size={hidden_size}, vocab_size={vocab_size}, num_heads={args.num_heads}")

    # 取 base_model 自己的 lm_head 权重，用于初始化每个 Medusa head
    # 这样训练起点就已经具备 base_model 同等的预测能力，而不是从随机权重学起
    base_lm_head_weight = base_model.lm_head.weight.data.clone()
    print(f"Base lm_head weight shape: {base_lm_head_weight.shape}, "
          f"std={base_lm_head_weight.std().item():.4f}")

    medusa_heads = MedusaHeadCollection(
        hidden_size, vocab_size, args.num_heads,
        init_lm_head_weight=base_lm_head_weight
    ).to(device)

    # 如果从断点继续训练
    if args.resume_from:
        print(f"Resuming from {args.resume_from} ...")
        state = torch.load(
            os.path.join(args.resume_from, "medusa_heads.pt"),
            map_location=device
        )
        medusa_heads.load_state_dict(state)

    trainable_params = sum(p.numel() for p in medusa_heads.parameters())
    print(f"Medusa heads params: {trainable_params:,}")

    # ---- 数据集 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. 加载完整的、不切分的数据集
    dataset = TextTokenDataset(
        args.data_file, tokenizer,
        max_length=args.max_length,
        max_tokens=args.max_tokens
    )

    # 2. 训练和验证都直接使用这套完整的数据
    train_ds = dataset
    val_ds = dataset

    # 3. 构建 DataLoader
    # 训练集开启 shuffle=True 增加随机性
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    # 验证集保持 shuffle=False，用于快速评估当前模型的收敛状态
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Using full dataset for both training and validation.")
    print(f"Total chunks: {len(dataset)}")

    # ---- 优化器和调度器 ----
    optimizer = torch.optim.AdamW(
        medusa_heads.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    total_steps   = len(train_loader) * args.num_epochs
    warmup_steps  = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ---- 训练循环 ----
    print(f"\nStart training: {args.num_epochs} epochs, "
          f"{len(train_loader)} steps/epoch")

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(args.num_epochs):
        medusa_heads.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for step, input_ids in enumerate(train_loader):
            input_ids = input_ids.to(device)

            # base model forward（no_grad，不更新base model）
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True
                )
                hidden = get_normed_hidden(base_model, outputs).detach()  # [B, seq, hidden]，已norm

            # Medusa heads forward + backward
            loss, accs = compute_loss_and_acc(
                hidden, medusa_heads, input_ids, args.num_heads
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(medusa_heads.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            # 打印进度
            if (step + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                avg_loss = epoch_loss / (step + 1)
                acc_str  = " | ".join([f"h{k}={a:.3f}" for k, a in enumerate(accs)])
                print(
                    f"Epoch {epoch+1}/{args.num_epochs} | "
                    f"Step {step+1}/{len(train_loader)} | "
                    f"Loss={avg_loss:.4f} | "
                    f"Acc: {acc_str} | "
                    f"LR={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.1f}s"
                )

        # ---- Epoch 结束：验证 ----
        val_loss, val_accs = evaluate(
            base_model, medusa_heads, val_loader, device, args.num_heads
        )
        acc_str = " | ".join([f"h{k}={a:.3f}" for k, a in enumerate(val_accs)])
        print(f"\n[Val] Epoch {epoch+1} | Loss={val_loss:.4f} | Acc: {acc_str}")

        # 保存逻辑：只保留一份权重，不再每个epoch都存
        if args.save_strategy == "last":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                medusa_heads.save(os.path.join(args.output_dir, "best"))
                print(f"[Save] Best model saved (val_loss={val_loss:.4f})")
        else:  # "last"：每个epoch结束都覆盖保存，最终留下的就是最后一个epoch的权重
            best_val_loss = min(best_val_loss, val_loss)
            medusa_heads.save(os.path.join(args.output_dir, "last"))
            print(f"[Save] Last model saved (val_loss={val_loss:.4f})")

    final_dir = os.path.join(args.output_dir, args.save_strategy)
    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {final_dir}")


# ==========================================
# 验证：统计每个 head 在压缩数据集上的 top-1 命中率
# ==========================================

def evaluate_on_compression_data(args):
    """
    在你的压缩数据集上评估每个 head 的 top-1 命中率和 rank 分布。
    这个函数帮你判断用几个 head 合适。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {args.model_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, device_map=device
    ).eval()
    if args.lora_path:
        base_model = PeftModel.from_pretrained(
            base_model, args.lora_path, torch_dtype=torch.float32
        )
        base_model = base_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading Medusa heads from {args.medusa_heads_dir} ...")
    medusa_heads = MedusaHeadCollection.load(args.medusa_heads_dir, device)
    medusa_heads.eval()

    with open(args.data_file, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text, add_special_tokens=True)[:10000]
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    print(f"\nEvaluating on {len(tokens)} tokens ...")

    with torch.no_grad():
        outputs = base_model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True
        )
        hidden    = get_normed_hidden(base_model, outputs)    # [1, seq, hidden]，已norm
        base_logits = outputs.logits             # [1, seq, vocab]

        all_head_logits = medusa_heads(hidden)   # list of [1, seq, vocab]

    # 统计每个 head 的 top-1 命中率和 rank 分布
    print(f"\n{'='*55}")
    print(f"Head-by-head 统计（压缩数据集）")
    print(f"{'='*55}")

    seq_len = input_ids.size(1)

    # head -1 是 base model 本身
    all_logits = [base_logits] + all_head_logits
    labels = ["base_model"] + [f"medusa_head_{k}" for k in range(len(all_head_logits))]

    for k, (logits, label) in enumerate(zip(all_logits, labels)):
        offset = k  # base model 预测 offset=1（下一个），head_k 预测 offset=k+1
        # 真正用于计算rank的offset
        pred_offset = k + 1  # 第k个输出预测第k+1步后的token

        if seq_len <= pred_offset:
            continue

        pred_logits = logits[0, :-pred_offset, :]   # [seq-offset, vocab]
        targets     = input_ids[0, pred_offset:]    # [seq-offset]

        # top-1 命中率
        top1  = pred_logits.argmax(dim=-1)
        acc   = (top1 == targets).float().mean().item()

        # rank 分布（取 top-32）
        _, topk_indices = torch.topk(pred_logits, k=32, dim=-1)  # [seq-offset, 32]
        expanded_targets = targets.unsqueeze(-1)
        matches    = (topk_indices == expanded_targets)
        found      = matches.any(dim=-1)
        rank_idx   = matches.int().argmax(dim=-1)
        ranks      = torch.where(found, rank_idx + 1, torch.tensor(33, device=device))

        rank1_rate  = (ranks == 1).float().mean().item()
        rank2_4     = ((ranks >= 2) & (ranks <= 4)).float().mean().item()
        rank5_32    = ((ranks >= 5) & (ranks <= 32)).float().mean().item()
        rank_miss   = (ranks > 32).float().mean().item()

        print(f"\n{label} (预测 +{pred_offset} 步):")
        print(f"  top-1 命中率 : {acc:.3f} ({acc*100:.1f}%)")
        print(f"  rank=1       : {rank1_rate:.3f}")
        print(f"  rank 2-4     : {rank2_4:.3f}")
        print(f"  rank 5-32    : {rank5_32:.3f}")
        print(f"  rank >32     : {rank_miss:.3f}")

    print(f"\n{'='*55}")
    print("建议：top-1命中率>=40%的head都值得使用")
    print("如果head_3的命中率已经很低，考虑只用2-3个head")


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # ---- 训练子命令 ----
    train_parser = subparsers.add_parser("train", help="训练 Medusa heads")
    train_parser.add_argument("--model_name",   type=str, required=True,
                               help="base model，如 Qwen/Qwen1.5-0.5B")
    train_parser.add_argument("--lora_path",    type=str, default="",
                               help="LoRA 权重路径（可选）")
    train_parser.add_argument("--data_file",    type=str, required=True,
                               help="训练数据文件（纯文本）")
    train_parser.add_argument("--output_dir",   type=str, default="./medusa_heads",
                               help="保存目录")
    train_parser.add_argument("--num_heads",    type=int, default=4,
                               help="Medusa head 数量，推荐 3-5")
    train_parser.add_argument("--num_epochs",   type=int, default=3)
    train_parser.add_argument("--batch_size",   type=int, default=4)
    train_parser.add_argument("--max_length",   type=int, default=512,
                               help="每个训练样本的 token 长度")
    train_parser.add_argument("--max_tokens",   type=int, default=2_000_000,
                               help="最多使用多少 token 训练")
    train_parser.add_argument("--lr",           type=float, default=1e-3)
    train_parser.add_argument("--log_every",    type=int, default=50)
    train_parser.add_argument("--resume_from",  type=str, default="",
                               help="从某个 checkpoint 继续训练")
    train_parser.add_argument("--save_strategy", type=str, default="best",
                               choices=["best", "last"],
                               help="'best'=只保存验证loss最低的一份；"
                                    "'last'=只保存最后一个epoch的一份（覆盖式）")

    # ---- 评估子命令 ----
    eval_parser = subparsers.add_parser("eval", help="评估 Medusa heads 命中率")
    eval_parser.add_argument("--model_name",        type=str, required=True)
    eval_parser.add_argument("--lora_path",         type=str, default="")
    eval_parser.add_argument("--data_file",         type=str, required=True)
    eval_parser.add_argument("--medusa_heads_dir",  type=str, required=True,
                              help="训练好的 Medusa heads 目录")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate_on_compression_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
