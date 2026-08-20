# LLMzip-code

###2. Rank Coding

Run rank-based compression and decompression verification:

```bash
# Basic run without LoRA
python -m compression.rank_coding \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --input_file "data/sample.txt" \
    --compressed_file "outputs/compressed.bin" \
    --output_file "outputs/decompressed.txt" \
    --dtype "fp16"

# Run with custom LoRA weights
python -m compression.rank_coding \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --lora_path "checkpoints/qwen_0.5b_lora" \
    --input_file "data/sample.txt" \
    --compressed_file "outputs/compressed.bin" \
    --output_file "outputs/decompressed.txt" \
    --dtype "fp16"
