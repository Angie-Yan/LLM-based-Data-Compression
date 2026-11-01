# LLMzip-code

## Environmental Setup

Clone the repository:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

Setup environment:

```bash
conda env create -f environment.yml
conda activate llm-dc
```

## Data Preparation

We evaluate our compression framework on both general text and source code benchmarks:

### Plain-Text Dataset (Wikipedia)
For text compression, we use the preprocessed **Simple English Wikipedia** dataset (Version 2) available on Kaggle:
- **Download:** [Plain-text Wikipedia SimpleEnglish (Kaggle)](https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish/versions/2)

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

## Pretrained MSMC Heads

One head set (K = 5 heads, FP16, ~1.4 GB) per domain, to train them yourself:

```bash
python Training/train_Mhead_shared.py train \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --train_files @data/<your_train_dataset_path> --test_files @data/<your_test_dataset_path> \
    --val_num_files 5 --num_heads 5 --max_length 1024 --batch_size 4 --bf16 \
    --output_dir ./MSMC_heads
```


## Data Compression

### Arithmetic Coding

    --batch_size 256
```
