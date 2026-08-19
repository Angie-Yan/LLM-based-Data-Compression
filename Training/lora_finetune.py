# import sys
#
# from blackwell.test_llama32_sft import max_seq_length
# from examples.research_projects.layer_skip.scripts.config import per_device_train_batch_size
# from examples.research_projects.layer_skip.scripts.config import gradient_accumulation_st
import os;

from fontTools.unicodedata import block
from sympy import false

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PEFT_DISABLE_CREATE_MODEL_CARD"] = "1"
from trl import SFTTrainer, SFTConfig
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import pprint
# from typing import List
import torch
# import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
# from utils.finetune_utils import CastOutputToFloat, print_trainable_parameters
# from huggingface_hub import login
# from transformers import BitsAndBytesConfig
import unsloth
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel   # FastVisionModel for LLMs
import warnings
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

from loss_function import CompressionLoss, CustomSFTTrainer
import matplotlib.pyplot as plt; import pandas as pd
import random



#🍅 Unsloth Version
def finetune(model_path,
             save_path,
             # dataset_path="/cpfs/user/liuyihao1/yanzhe/LLM-DC/data/wiki_00",
             block_size=512,
             epochs=10,
             r=16,
             # learning_rate=5e-6,
             # batch_size=4,
             max_seq_length = 1024,
             lora_rank = 16,
             load_method = "2",
             ):

        # 使用 FastLanguageModel.from_pretrained 方法加载预训练模型和分词器
        model, tokenizer = FastLanguageModel.from_pretrained(
            # model_name = "Qwen/Qwen2.5-3B-Instruct", # 指定要加载的模型名称
            # model_name = "/home/jovyan/llms/Qwen2.5-3B-Instruct", # 模型路径
            model_name="/root/yanzhe/yanzhe/LLM-DC/Qwen2.5-0.5B-Instruct",
            max_seq_length=max_seq_length,  # 设置模型的最大序列长度
            load_in_4bit=True,  # 设置为 True 时，模型以 4 位精度加载以减少内存占用；设置为 False 时使用float16
            dtype=torch.float16,
            fast_inference=False,  # 启用 vLLM 快速推理以加速模型推理过程
            max_lora_rank=lora_rank,  # 设置 LoRA 的最大秩
            gpu_memory_utilization=0.5, # 设置 GPU 内存利用率，如果内存不足，可以降低这个值
            trust_remote_code=True,
            use_cache=False
        )
        # tokenizer = AutoTokenizer.from_pretrained(model_path,
        #                                           use_fast=False,
        #                                           local_files_only = True,
        #                                           trust_remote_code=True)

        # tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = model.config.eos_token_id
        tokenizer.model_max_length = max_seq_length
        # model._saved_temp_tokenizer = tokenizer  # 手动补充属性

        # Loading Lora
        model = FastLanguageModel.get_peft_model(
            model,  # import model
            r=lora_rank,  # set lora rank
            lora_alpha=lora_rank,  # lora 缩放因子
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 训练权重
            use_gradient_checkpointing="unsloth",  # 启用梯度检查点技术，以支持长上下文微调并减少内存占用
            random_state=2025,  # 设置随机种子，确保实验的可重复性
        )
        print("model loaded")

        # # 原来的数据加载方式：按照语句自动切分
        # train_dataset = load_dataset("text", data_files={"train": dataset_path})["train"]
        # # 取前100000个token作为训练集
        # train_dataset = train_dataset.select(range(100000))

        # 新的dataset加载方式：按blocksize切分
        def prepare_lm_dataset(
                dataset_path,
                tokenizer,
                max_tokens=1000000,
                block_size = block_size,
                load_method="3",
        ):
            """
            load_method:
                - "1": 顺序取前 max_tokens 个 token（你原来的逻辑）
                - "2": 全局随机抽取 batch_size 段，每段 block_size个 token
                - "3": 全局数据 finetune，适合比较小的数据（20M之内）
            """

            # load dataset text
            raw_dataset = load_dataset("text", data_files = {"train":dataset_path})["train"]
            def tokenize_fn(examples):
                out = tokenizer(
                    examples["text"],
                    truncation=False,
                    add_special_tokens=False
                )
                return {"input_ids": out["input_ids"]}

            # tokenize text
            tokenized_dataset = raw_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns = raw_dataset.column_names
            )

            # flatten all the token
            all_tokens = []
            for example in tokenized_dataset:
                all_tokens.extend(example["input_ids"])
            total_tokens = len(all_tokens)
            print(f"[INFO] Total tokens = {total_tokens}")

            # load_method 1
            if load_method == "1":
                print("[INFO] Using load_method=1 (sequential)")
                use_tokens = all_tokens[:max_tokens]

            # load_method 2
            elif load_method == "2":
                print("[INFO] Using load_method=2 (random segments)")
                if total_tokens < block_size:
                    raise ValueError(f"the dataset token is less than ({total_tokens}), cannot take block_size={block_size}")
                batch_size = max_tokens // block_size
                use_tokens = []
                for _ in range(batch_size):
                    start = random.randint(0, total_tokens - block_size)
                    seg = all_tokens[start: start + block_size]
                    use_tokens.extend(seg)

                print(f"[method2] 抽取了 {batch_size} 段 × {block_size} tokens，共 {len(use_tokens)} tokens")

            # load_method 3
            elif load_method == "3":
                print("[INFO] Using load_method=3 (use ALL tokens, full dataset)")
                use_tokens = all_tokens  # use all token

            else:
                raise ValueError(f"Invalid load_method = {load_method}")

            # 切分 block size
            num_blocks = len(use_tokens) // block_size
            lm_input_ids = [
                use_tokens[i * block_size: (i + 1) * block_size]
                for i in range(num_blocks)
            ]

            lm_dataset = Dataset.from_list([
                {"input_ids": block, "labels": block.copy()}
                for block in lm_input_ids
            ])

            return lm_dataset


        train_dataset = prepare_lm_dataset(
            dataset_path="/root/yanzhe/yanzhe/LLM-DC/data/wiki_05",tokenizer=tokenizer,
            block_size=512, load_method = load_method,
        )

        print("Dataset loaded")
        print(f"训练样本数: {len(train_dataset)}")

        model.print_trainable_parameters()  # 打印确认，看看lora是否正常挂上

        # after get_peft_model
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

        # 再次确认
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Trainable param count = {len(trainable)}")
        print(trainable[:50])

        # trainer = SFTTrainer(
        #     model = model,
        #     train_dataset = train_dataset,
        #     args = SFTConfig(
        #         per_device_train_batch_size = 16,
        #         gradient_accumulation_steps = 1,
        #         warmup_steps = 50,
        #         max_steps = 8000,
        #         learning_rate = 1e-4,
        #         lr_scheduler_type = "cosine",
        #         logging_steps = 10,
        #         output_dir = "outputs",
        #         optim = "adamw_torch_fused",
        #         seed = 3407,
        #         max_grad_norm = 1.0,
        #         fp16 = False,
        #         bf16 = True,
        #         half_precision_backend="auto"
        #     ),

        # use new loss function here:
        criterion = CompressionLoss(alpha=1.0, beta=1.0, margin=1.5, omega=0.0, gamma=0.0, use_focal=True)
        # # 自定义trainer
        trainer = CustomSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=16,
                gradient_accumulation_steps=1,
                warmup_steps=50,
                max_steps=8000,
                learning_rate=1e-4,
                lr_scheduler_type="cosine",
                logging_steps=10,
                output_dir="outputs",
                optim="adamw_torch_fused",
                seed=3407,
                max_grad_norm=1.0, #防止爆炸
                fp16=False,
                bf16=True,
                half_precision_backend="auto"
            ),
            criterion=criterion,  # the new loss funtion
         )

        # 检测函数，确保成功挂上lora：
        # trainable = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         trainable.append(name)
        # print(f"Trainable param count = {len(trainable)}")
        # print(trainable[:50])  # 打印前 50 个可训练参数名字

        for n, p in model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
                print(n, p.requires_grad)

        trainer.train()
        # ========== 绘制 Loss 曲线 ==========
        # losses = []
        # steps = []
        #
        # for log in trainer.state.log_history:
        #     if "loss" in log and "step" in log:
        #         losses.append(log["loss"])
        #         steps.append(log["step"])
        #
        # if len(losses) > 0:
        #     # 保存为 CSV
        #     os.makedirs("outputs", exist_ok=True)
        #     df = pd.DataFrame({"step": steps, "loss": losses})
        #     csv_path = os.path.join("outputs", f"loss_log_{epochs}_r{r}.csv")
        #     df.to_csv(csv_path, index=False)
        #     print(f"Loss 日志已保存到 {csv_path}")
        #
        #     # 绘制曲线
        #     plt.figure(figsize=(8, 5))
        #     plt.plot(steps, losses, label="Training Loss", linewidth=1.8)
        #     plt.xlabel("Step")
        #     plt.ylabel("Loss")
        #     plt.title(f"Finetuning Loss Curve (epochs={epochs}, r={r})")
        #     plt.legend()
        #     plt.grid(True)
        #     fig_path = os.path.join("loss_function_graph", f"loss_curve_{epochs}_r{r}.png")
        #     plt.savefig(fig_path)
        #     plt.close()
            # print(f"Loss curve has been saved to {fig_path}")

        trainer.save_model(save_path + f"_{epochs}_r{r}")

        del model
        del trainer
        torch.cuda.empty_cache()

        print("Finished finetuning")


if __name__ == "__main__":
    # login("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")

    finetune_list = [
        "/root/yanzhe/yanzhe/LLM-DC/Qwen2.5-0.5B-Instruct"
    ]
    epoch_list = [128]

    for model in finetune_list:
        for e in epoch_list:
            finetune(
                model, 
                save_path=f"finetuned_models/h100/Qwen0.5b/wiki_05_newloss",
                # dataset_path="/cpfs/user/liuyihao1/yanzhe/LLM-DC/data/books_large_p2.txt",
                block_size=512, # 每个样本序列长度
                epochs=e, 
                r=16, # lora rank，低秩矩阵大小
                # learning_rate=1e-4,
                # batch_size=16,
                max_seq_length=1024,
                lora_rank= 16,
                load_method="3",
            )
