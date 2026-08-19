import os
# 设置 Hugging Face 镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import glob
from datasets import load_dataset

# ============================
# 配置
# ============================

OUTPUT_PATH = "/root/yanzhe/yanzhe/LLM-DC/data/mixed_dataset_50mb.txt"
ROCKSDB_PATH = "/root/yanzhe/yanzhe/rocksdb"
TARGET_SIZE_MB = 10  # 收集到的数据集大小
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

# 随机混合数据时的权重（你可自己调整）
WEIGHTS = {
    "mbpp": 1,
    "humaneval": 1,
    "humaneval_x": 1,
    "rocksdb": 2,  # 本地代码多，权重稍大
}

# ============================
# 工具函数
# ============================

def load_mbpp():
    ds = load_dataset("mbpp", split="train")
    samples = []
    for item in ds:
        if item.get("code"):
            samples.append(item["code"])
    print(f"[MBPP] Loaded {len(samples)} samples.")
    return samples

# def load_humaneval():
#     # ds = load_dataset("openai/openai_humaneval", split="test")
#     ds = load_dataset("openai_humaneval", split="test")
#     samples = []
#     for item in ds:
#         samples.append(item["prompt"] + "\n" + item["canonical_solution"])
#     print(f"[Humaneval] Loaded {len(samples)} samples.")
#     return samples
#
# def load_humaneval_x():
#     ds = load_dataset("zai-org/humaneval-x", split="test")
#     samples = []
#     for item in ds:
#         for lang, code in item["solutions"].items():
#             if code:
#                 samples.append(f"# Language: {lang}\n" + code)
#     print(f"[Humaneval-X] Loaded {len(samples)} samples.")
#     return samples

def load_rocksdb():
    CODE_EXT = ["*.cc", "*.h", "*.cpp", "*.c", "*.hpp"]
    files = []
    for ext in CODE_EXT:
        files += glob.glob(os.path.join(ROCKSDB_PATH, "**", ext), recursive=True)

    code_pieces = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                code_pieces.append(fp.read())
        except:
            pass

    print(f"[rocksdb] Loaded {len(code_pieces)} code files.")
    return code_pieces


# ============================
# 主函数：混合生成数据集
# ============================

def main():
    print("Loading datasets...")

    mbpp_data = load_mbpp()
    # humaneval_data = load_humaneval()
    # humaneval_x_data = load_humaneval_x()
    rocksdb_data = load_rocksdb()

    sources = [
        ("mbpp", mbpp_data),
        # ("humaneval", humaneval_data),
        # ("humaneval_x", humaneval_x_data),
        ("rocksdb", rocksdb_data),
    ]

    # 权重展开机制
    weighted_sources = []
    for name, data in sources:
        weighted_sources += [(name, data)] * WEIGHTS[name]

    print("\nStart mixing ...")
    output = []
    size = 0

    while size < TARGET_SIZE_BYTES:
        src_name, src_data = random.choice(weighted_sources)
        if len(src_data) == 0:
            continue

        sample = random.choice(src_data)
        output.append(f"\n# ===== Source: {src_name} =====\n")
        output.append(sample)
        output.append("\n\n")

        size = sum(len(s) for s in output)

        print(f"Current size: {size/1024/1024:.2f} MB", end="\r")

    print("\n\nWriting output...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fp:
        fp.write("".join(output))

    print(f"\n========== DONE ==========")
    print(f"Final dataset size: {size/1024/1024:.2f} MB")
    print(f"Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
