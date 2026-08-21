# LLMzip-code

## Environmental Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

```bash
conda env create -f environment.yml
conda activate llm-dc
```

## Data Preparation

We evaluate our compression framework on both general text and source code benchmarks:

### Plain-Text Dataset (Wikipedia)
For text compression, we use the preprocessed **Simple English Wikipedia** dataset available on Kaggle:
- **Download:** [Plain-text Wikipedia SimpleEnglish (Kaggle)](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish)

Please download the dataset manually and place the extracted text file under the `data/` directory:
```text
data/
└── wiki_simpleenglish.txt
```

### Code Dataset

For code compression, we provide a zero-configuration script to automatically download and construct a 1MB mixed multi-language code dataset (sampling from MBPP, HumanEval, HumanEval-X, and C++ repositories). Simply run:

```bash
python scripts/Code_Data_Collection.py
```

## Data Compression

### Arithmetic Coding

Arithmetic Coding (AC) provides near-optimal entropy coding based on the probability distribution predicted by the language model. This script supports both compression and decompression pipelines with optional global mask space reduction.

#### 1. Compress
Compress a raw text dataset into a binary file:

```bash
# Compress plain-text dataset
python Compression/Pure_AC.py \
    --mode "compress" \
    --model_name "Qwen2.5-0.5B" \
    --input_path "data/<path_to_your_dataset>" \
    --output_path "out.bin" \
    --context_length 1024 \
    --batch_size 256 \
    --print_results
```

#### Decompress

Reconstruct the original text from the compressed binary file:

```bash
python Compression/Pure_AC.py \
    --mode "decompress" \
    --model_name "Qwen2.5-0.5B" \
    --input_path "out.bin" \
    --context_length 1024 \
    --batch_size 256 \
    --print_results
```

### Rank Coding

Run rank-based compression and decompression verification:

```bash
# Basic run without LoRA
python Compression/Fast_Rank_Compression.py \
    --model_name "Qwen2.5-0.5B" \
    --input_file "data/<path_to_your_dataset>" \
    --dtype "fp16" \
    --batch_size 256
```

Run with custom LoRA weights
```bash
python Compression/Fast_Rank_Compression.py \
    --model_name "Qwen2.5-0.5B" \
    --lora_path "checkpoints/qwen_0.5b_lora" \
    --input_file "data/<path_to_your_dataset>" \
    --dtype "fp16" \
    --batch_size 256
```

### Speculative compression (SDMC)



### Medusa Compression

Medusa-accelerated rank compression utilizes parallel decoding heads to speed up the compression and decompression pipeline.

```bash
# Standard run with pre-trained Medusa heads
python Compression/Medusa_Compression.py \
    --model_name "Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --input_file "data/<path_to_your_dataset>" \
    --dtype "fp16" \
    --batch_size 256
```

Run with a subset of active heads (e.g., test 3 active heads (total heads > 3) without retraining)

```bash
python Compression/Medusa_Compression.py \
    --model_name "Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --num_active_heads 3 \
    --input_file "data/<path_to_your_dataset>" \
    --dtype "fp16" \
    --batch_size 256
```
