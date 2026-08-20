# LLMzip-code

### 2. Rank Coding

Run rank-based compression and decompression verification:

```bash
# Basic run without LoRA
python -m compression.rank_coding \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```

Run with custom LoRA weights
```bash
python -m compression.rank_coding \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --lora_path "checkpoints/qwen_0.5b_lora" \
    --input_file "data/sample.txt" \
    --dtype "fp16" \
    --batch_size 256
```
    

### 4. Medusa Compression

Medusa-accelerated rank compression utilizes parallel decoding heads to speed up the compression and decompression pipeline.

```bash
# Standard run with pre-trained Medusa heads
python -m compression.medusa_compression \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --input_file "data/sample.txt" \
    --compressed_file "outputs/medusa_compressed.bin" \
    --output_file "outputs/medusa_decompressed.txt" \
    --dtype "fp16"
```

Run with a subset of active heads (e.g., test 3 active heads without retraining)

```bash
python -m compression.medusa_compression \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --medusa_heads_dir "checkpoints/medusa_heads_qwen0.5b" \
    --num_active_heads 3 \
    --input_file "data/sample.txt" \
    --dtype "fp16"
```
