import torch
import torch.nn as nn
import torch.nn.functional as F
import unsloth
from trl import SFTTrainer, SFTConfig


class CompressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, margin=1.0, omega=0.0, gamma=0.0, use_focal=False):
        """
        :param alpha: CE / Focal loss 的权重 (保底，维持模型语言能力)
        :param beta: Margin loss 的权重 (主攻，强制提升 Top-1 命中率)
        :param margin: Margin 的安全阈值 (通常设为 1.0 - 2.0)
        :param omega: Entropy loss 的权重 (防止过度自信，通常较小或 0)
        :param gamma: KL loss 的权重 (用于知识蒸馏)
        :param use_focal: 是否启用 Focal Loss 替代标准 CE Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.omega = omega
        self.gamma = gamma
        self.use_focal = use_focal

    def forward(self, logits, labels, base_logits=None):
        """
        :param logits: outputs of LLM model, size [batch, vocab_size]
        :param labels: truth token id, size [batch]
        :param base_logits: outputs of base model (optional, used for KL loss)
        """
        dtype = logits.dtype
        device = logits.device
        batch_size, vocab_size = logits.size()

        # ==========================================
        # 1. CE Loss (或 Focal Loss)
        # ==========================================
        if self.use_focal:
            # 引入 Focal 机制：降低已经命中（高概率）样本的权重，专注困难样本
            ce_loss_none = F.cross_entropy(logits, labels, reduction="none")
            pt = torch.exp(-ce_loss_none)
            loss_CE = (((1 - pt) ** 4.0) * ce_loss_none).mean()
        else:
            # 基础 CE Loss
            loss_CE = F.cross_entropy(logits, labels, reduction="mean")

        # ==========================================
        # 2. 绝对命中定制损失 (Top-1 Margin Loss)
        # ==========================================
        correct_logits = logits[torch.arange(batch_size), labels].unsqueeze(1)

        # 找出最强竞争者 (除正确 token 外的最大 logit)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), False)
        # 将正确 token 的位置用负无穷填补，使其在求 max 时被忽略
        neg_logits = logits.masked_fill(~mask, float('-inf'))
        max_neg_logits, _ = neg_logits.max(dim=1, keepdim=True)

        # 计算 Margin Loss: max(0, margin + 最强错误 logit - 正确 logit)
        # 只要正确答案比第二名高出 margin 的距离，Loss 就是 0，节省模型容量
        loss_margin = F.relu(self.margin + max_neg_logits - correct_logits).mean()

        # ==========================================
        # 2.1 BPR loss (比margin loss更有针对性)
        # ==========================================
        # 取出正确 token 的 logits: [B, 1]
        # correct_logits = logits[torch.arange(batch_size), labels].unsqueeze(1)
        #
        # # 找出最强竞争者 (除正确 token 外的最大 logit)
        # mask = torch.ones_like(logits, dtype=torch.bool)
        # mask.scatter_(1, labels.unsqueeze(1), False)
        # neg_logits = logits.masked_fill(~mask, float('-inf'))
        # max_neg_logits, _ = neg_logits.max(dim=1, keepdim=True)
        #
        # # BPR Loss 核心逻辑：
        # # 我们希望 (correct_logits - max_neg_logits) 越大越好
        # # 使用 logsigmoid: -log(sigmoid(correct - max_neg))
        # # 这等价于 Softplus(max_neg - correct)
        #
        # # 为了更极致的 Top-1 命中，我们引入一个缩放因子 lambda (或利用你的 margin)
        # # 这里的 self.margin 可以理解为对“差距”的最低要求
        # diff = correct_logits - max_neg_logits
        #
        # # 方案 A：标准 BPR (无硬性阈值，持续拉开差距)
        # # loss_margin = -F.logsigmoid(diff).mean()
        #
        # # # 方案 B：带 Margin 的 BPR (更适合你的需求，强制拉开至少 margin 的距离)
        # # # 当差距小于 margin 时，梯度极大；当差距远大于 margin 时，梯度消失，节省模型容量
        # loss_margin = -F.logsigmoid(diff - self.margin).mean()


        # ==========================================
        # 3. Entropy Loss (保留你的原逻辑，作为正则项)
        # ==========================================
        loss_entropy = torch.zeros((), dtype=dtype, device=device)
        if self.omega > 0:
            probs = F.softmax(logits, dim=-1)
            eps = torch.tensor(1e-12, dtype=dtype, device=device)
            loss_entropy = -(probs * torch.log(probs + eps)).sum(dim=1).mean()

        # ==========================================
        # 4. KL Loss (保留你的原逻辑)
        # ==========================================
        loss_KL = torch.zeros((), dtype=dtype, device=device)
        if base_logits is not None and self.gamma > 0:
            probs = F.softmax(logits, dim=-1)
            base_probs = F.softmax(base_logits, dim=-1)
            loss_KL = F.kl_div(probs.log(), base_probs, reduction="batchmean")

        # ==========================================
        # 5. Final Loss
        # ==========================================
        loss = self.alpha * loss_CE + self.beta * loss_margin + self.omega * loss_entropy + self.gamma * loss_KL

        return loss, {"CE": loss_CE.detach().float().item(),
                      "Margin": loss_margin.detach().float().item(),
                      "Entropy": loss_entropy.detach().float().item(),
                      "KL": loss_KL.detach().float().item()}



class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, criterion=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion

    def compute_loss(self, model, inputs, return_outputs=False, **kwrags):
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab]
        labels = inputs["labels"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        #[N, vocab]
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # 只保留非 -100 标签（pad 忽略位）
        active_mask = shift_labels != -100
        active_logits = shift_logits[active_mask]
        active_labels = shift_labels[active_mask]

        loss, details = self.criterion(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss
