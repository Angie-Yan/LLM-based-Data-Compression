import os
import random
from datasets import load_dataset

# Configuration
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mixed_code_dataset.txt")
TARGET_SIZE_MB = 1
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

# Data mix sampling weights (can be adjusted)
WEIGHTS = {
    "mbpp": 1,
    "humaneval": 1,
    "humaneval_x": 1,
    "rocksdb": 1,
}


def load_mbpp():
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        samples = [item["code"] for item in ds if item.get("code")]
        print(f"[MBPP] Loaded {len(samples)} samples.")
        return samples
    except Exception as e:
        print(f"[MBPP] Failed to load: {e}")
        return []


def load_humaneval():
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        samples = [f"{item['prompt']}\n{item['canonical_solution']}" for item in ds]
        print(f"[Humaneval] Loaded {len(samples)} samples.")
        return samples
    except Exception as e:
        print(f"[Humaneval] Failed to load: {e}")
        return []


def load_humaneval_x():
    try:
        ds = load_dataset("THUDM/humaneval-x", "python", split="test")
        samples = [f"# Language: Python\n{item['declaration']}\n{item['canonical_solution']}" for item in ds if item.get("canonical_solution")]
        print(f"[Humaneval-X] Loaded {len(samples)} samples.")
        return samples
    except Exception as e:
        print(f"[Humaneval-X] Failed to load: {e}")
        return []


def load_rocksdb_online():
    """Load C++ code snippets online from CodeSearchNet C++ subset as a proxy for RocksDB-like C++ code."""
    try:
        ds = load_dataset("code_search_net", "cpp", split="train[:500]")
        samples = [item["whole_func_string"] for item in ds if item.get("whole_func_string")]
        print(f"[RocksDB/C++] Loaded {len(samples)} samples online.")
        return samples
    except Exception as e:
        print(f"[RocksDB/C++] Failed to load online: {e}")
        return []


def main():
    print("Fetching code datasets online...")

    sources = [
        ("mbpp", load_mbpp()),
        ("humaneval", load_humaneval()),
        ("humaneval_x", load_humaneval_x()),
        ("rocksdb", load_rocksdb_online()),
    ]

    # Filter out empty datasets
    active_sources = [(name, data) for name, data in sources if len(data) > 0]
    if not active_sources:
        raise RuntimeError("No datasets were successfully loaded. Please check your network connection.")

    weighted_sources = []
    for name, data in active_sources:
        weighted_sources += [(name, data)] * WEIGHTS[name]

    print("\nMixing and generating 1MB benchmark dataset...")
    output = []
    size = 0

    while size < TARGET_SIZE_BYTES:
        src_name, src_data = random.choice(weighted_sources)
        sample = random.choice(src_data)
        output.append(f"\n# ===== Source: {src_name} =====\n")
        output.append(sample)
        output.append("\n\n")

        size = sum(len(s.encode("utf-8")) for s in output)
        print(f"Current size: {size / (1024 * 1024):.2f} MB", end="\r")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
        fp.write("".join(output))

    print(f"\n\n========== DONE ==========")
    print(f"Final dataset size: {size / (1024 * 1024):.2f} MB")
    print(f"Output saved to relative path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
