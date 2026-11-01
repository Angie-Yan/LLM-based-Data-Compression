import torch
import torch.nn as nn
import torch.nn.functional as F
import unsloth
from trl import SFTTrainer, SFTConfig


class CompressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, margin=1.0, use_focal=False):
        """
        :param alpha: CE / Focal loss 的权重 (保底，维持模型语言能力)
        :param beta: Margin loss 的权重 (主攻，强制提升 Top-1 命中率)
        :param margin: Margin 的安全阈值 (通常设为 1.0 - 2.0)
        :param use_focal: 是否启用 Focal Loss 替代标准 CE Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
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

      
        # 1. CE Loss (或 Focal Loss)
        if self.use_focal:
            # 引入 Focal 机制：降低已经命中（高概率）样本的权重，专注困难样本
            ce_loss_none = F.cross_entropy(logits, labels, reduction="none")
            pt = torch.exp(-ce_loss_none)
            loss_CE = (((1 - pt) ** 4.0) * ce_loss_none).mean()
        else:
            # 基础 CE Loss
            loss_CE = F.cross_entropy(logits, labels, reduction="mean")

      
        # 2. 绝对命中定制损失 (Top-1 Margin Loss)
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

      
        # 3. Final Loss
        loss = self.alpha * loss_CE + self.beta * loss_margin

        return loss, {"CE": loss_CE.detach().float().item(),
                      "Margin": loss_margin.detach().float().item(),
                      }



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
