# The Invocation Bottleneck in LLM-Based Lossless Compression: What Can Be Removed and What Must Be Traded
Code for the paper The Invocation Bottleneck in LLM-Based Lossless Compression: What Can Be Removed and What Must Be Traded. This repository contains a GPU-resident LLM compression pipeline with two entropy coders (arithmetic coding and Rank-Tier coding), batch-invariant kernels that make FP16 encoding and decoding bit-exact, MSMC (multi-head single-model compression), and SDMC (speculative dual-model compression).

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

### Code Dataset

Built from RocksDB C/C++ sources plus a small share of MBPP Python. Test files are drawn from RocksDB top-level directories that are held out entirely; PyTorch C++ headers are used as additional training-only data. We do not redistribute the corpus: it is rebuilt from public sources with Scripts/build_code_corpus.py, which needs three inputs:
For code compression, we provide a zero-configuration script to automatically download and construct a 1MB mixed multi-language code dataset (sampling from MBPP, HumanEval, HumanEval-X, and C++ repositories). Simply run:

1. RocksDB (a plain clone is enough; RocksDB does not need to be built):

```bash
git clone https://github.com/facebook/rocksdb.git
git -C rocksdb checkout ac412b10955d5a1d3d99aff8edf94eae1e4a22d5
```

2. MBPP, loaded with Hugging Face datasets (mbpp). Without network access, pass a local copy with --mbpp_path mbpp.jsonl (.jsonl, .json or .parquet).

3. PyTorch C++ headers from the environment in environment.yml (torch <version>). They are used for training data only; a different torch version changes the training data, not the test set.

Rebuild the exact paper split from the released manifest:

```bash
python Scripts/Code_data_Collection.py --rocksdb_dir rocksdb \
    --from_manifest Scripts/splits/code_manifest.json \
    --extra_train_dirs $CONDA_PREFIX/lib/python3.10/site-packages/torch/include \
    --out_dir data/code_corpus
```

The manifest lists, in order, the source files and MBPP tasks that make up each test sample, and the SHA-256 of each sample. The script prints OK for every test sample it reproduces byte for byte, and warns if the RocksDB commit differs.

Build a new split (e.g. with another seed) by omitting --from_manifest:

```bash
python Scripts/build_code_corpus.py --rocksdb_dir rocksdb --out_dir data/code_corpus \
    --no_humaneval --split_by dir --seed 0 \
    --extra_train_dirs $CONDA_PREFIX/lib/python3.10/site-packages/torch/include
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

## Compress and Decompress

--round_size n is the number of positions decoded per backbone call: n = 1 uses the backbone alone (no heads); n = k + 1 additionally uses the first k MSMC heads.

```bash
COMMON="--model_name Qwen/Qwen2.5-0.5B-Instruct --medusa_heads_dir ./MSMC_heads \
  --dtype fp16 --attn_implementation eager --batch_invariant_attention --batch_invariant_linear \
  --batch_size 256 --round_size 4"

# MSMC + AC: encode and decode in one process, then verify byte-exact reconstruction
python Compression/MSMC_AC.py $COMMON \
    --input_file data/<your_dataset_path> --compressed_file compressed_ac.bin --output_file decoded_ac.txt

# MSMC + RC
python Compression/Medusa_Compression.py $COMMON \
    --input_file data/<your_dataset_path> --compressed_file compressed_rc.txt --output_file decoded_rc.txt
```

Separate-process decoding. Encoding and decoding can run in different processes; the decoder reloads the backbone and heads and reads only the compressed file:

```bash
python Compression/MSMC_AC.py $COMMON --mode encode --input_file F --compressed_file c.bin
python Compression/MSMC_AC.py $COMMON --mode decode --compressed_file c.bin \
    --output_file d.txt --input_file F          # --input_file is used only for verification
cmp F d.txt && sha256sum F d.txt
```

Important flags.

--batch_invariant_attention --batch_invariant_linear are required for lossless FP16 decoding with teacher-forced encoding. Without them, encoder and decoder logits can differ in the last bits.

--total_bits (AC only): probability quantization precision, default 2^24.
