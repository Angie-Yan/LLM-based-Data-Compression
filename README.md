# LLMzip-code

## Data Preparation

We evaluate our compression framework on both general text and source code benchmarks:

### 1. Plain-Text Dataset (Wikipedia)
For text compression, we use the preprocessed **Simple English Wikipedia** dataset available on Kaggle:
- **Download:** [Plain-text Wikipedia SimpleEnglish (Kaggle)](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish)

Please download the dataset manually and place the extracted text file under the `data/` directory:
```text
data/
└── wiki_simpleenglish.txt
```

### 2. Code Dataset

For code compression, we provide a zero-configuration script to automatically download and construct a 1MB mixed multi-language code dataset (sampling from MBPP, HumanEval, HumanEval-X, and C++ repositories). Simply run:

```bash
python scripts/prepare_code_data.py
```

## Data Compression

```bash
cd Compression
```

### 1. Arithmetic Coding

### 2. Rank Coding

Run rank-based compression and decompression verification:

```bash
# Basic run without LoRA
python Fast_Rank_Compression.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```

Run with custom LoRA weights
```bash
python Fast_Rank_Compression.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --lora_path "checkpoints/qwen_0.5b_lora" \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```

### 3. Speculative compression (SDMC)



### 4. Medusa Compression

Medusa-accelerated rank compression utilizes parallel decoding heads to speed up the compression and decompression pipeline.

```bash
# Standard run with pre-trained Medusa heads
python Medusa_Compression.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```

Run with a subset of active heads (e.g., test 3 active heads (total heads > 3) without retraining)

```bash
python Medusa_Compression.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --num_active_heads 3 \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```
