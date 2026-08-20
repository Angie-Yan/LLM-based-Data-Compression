"""
llm_ac_compress.py

Consolidated, Arithmetic-Coding-only version of the "Global Mask LLM Compression"
pipeline. This merges the relevant pieces from the original 5-file project
(args.py, encoding.py, main.py, prediction.py, utils.py) into a single file,
stripping out everything related to:
    - LoRA/VeRA adapter training
    - Model quantization
    - bitpacked / huffman / zstd encoding schemes

Only the Arithmetic Coding (AC) compression/decompression path is kept.

Usage:
    Compress:
        python llm_ac_compress.py --mode compress \
            --input_path data/text8 \
            --output_path out.bin \
            --model_name gpt2 \
            --first_n_tokens 2000

    Decompress:
        python llm_ac_compress.py --mode decompress \
            --input_path out.bin \
            --output_path reconstructed.txt
"""

import argparse
import json
import math
import os
import struct
import time
from datetime import datetime
from itertools import chain
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from pyroaring import BitMap


# =============================================================================
# Section 1: CLI Args  (from args.py, trimmed to AC-only / no adapters/quant)
# =============================================================================

def get_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Global Mask AC Compression Experiment")

    # path related
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True,
                         help="Mode: compress or decompress")
    parser.add_argument("--input_path", type=str,
                        help="Input path: For compress mode, dataset path. For decompress mode, compression file path.")
    parser.add_argument("--output_path", type=str, help="Output path: For compress mode, compression file path. "
                                                          "For decompress mode, reconstruction text file path.")
    parser.add_argument("--text_input", type=str, required=False, help="Direct text input for LLM inference.")

    # model related
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path",
    )

    # inference related
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum context length")
    parser.add_argument("--retain_tokens", type=int, default=100,
                         help="Tokens retained when context length exceeded (only with KV cache)")
    parser.add_argument("--first_n_tokens", type=int, default=200000, help="Number of tokens to compress")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for compression")
    parser.set_defaults(use_kv_cache=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for LLM inference")

    # compression related (AC only)
    parser.add_argument("--reduce_tokens", action="store_true", help="Restrict token space to tokens seen in data")
    parser.add_argument("--no_reduce_tokens", dest="reduce_tokens", action="store_false",
                         help="Disable token space restriction")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--engine", type=str, choices=["transformer"], default="transformer",
                         help="Inference engine to use")

    # other
    parser.add_argument("--print_results", action="store_true", help="Print detailed results")

    args = parser.parse_args()

    # encoding is fixed to AC in this stripped-down version
    args.encoding = "AC"

    # Detect seq2seq models (T5)
    args.is_seq2seq = "t5" in args.model_name.lower()

    # Disable KV cache for T5
    if args.is_seq2seq:
        if args.use_kv_cache:
            print("KV cache disabled for T5 models.")
        args.use_kv_cache = False

    return args


# =============================================================================
# Section 2: Arithmetic Coding core (from encoding.py, AC parts only)
# =============================================================================

class ArithmeticCoderBase(object):
    """Shared state/range logic for the arithmetic encoder and decoder."""

    def __init__(self, statesize):
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << self.STATE_SIZE
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        self.MAX_TOTAL = self.MIN_RANGE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1

        self.low = 0
        self.high = self.MASK

    def update(self, cumul, symbol):
        low = self.low
        high = self.high
        range_ = high - low + 1

        total = cumul[-1].item()
        symlow = cumul[symbol].item()
        symhigh = cumul[symbol + 1].item()

        newlow = low + symlow * range_ // total
        newhigh = low + symhigh * range_ // total - 1
        self.low = newlow
        self.high = newhigh

        # While the highest bits are equal
        while ((self.low ^ self.high) & self.TOP_MASK) == 0:
            self.shift()
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1

        # While low=01(...) and high=10(...)
        while (self.low & ~self.high & self.SECOND_MASK) != 0:
            self.underflow()
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


class ArithmeticEncoder(ArithmeticCoderBase):
    def __init__(self, statesize, bitout):
        super(ArithmeticEncoder, self).__init__(statesize)
        self.output = bitout
        self.num_underflow = 0

    def write(self, cumul, symbol):
        self.update(cumul, symbol)

    def finish(self):
        self.output.write(1)

    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        self.output.write(bit)
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1


class ArithmeticDecoder(ArithmeticCoderBase):
    def __init__(self, statesize, bitin):
        super(ArithmeticDecoder, self).__init__(statesize)
        self.input = bitin
        self.code = 0
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()

    def read(self, cumul, alphabet_size):
        total = cumul[-1].item()
        range_ = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_

        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle

        symbol = start
        self.update(cumul, symbol)
        return symbol

    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()

    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()

    def read_code_bit(self):
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp


class BitOutputStream:
    """Collects single bits written by the encoder."""

    def __init__(self):
        self.bits = []

    def write(self, bit: int):
        self.bits.append(bit & 1)

    def get_bits(self):
        return self.bits


class BitInputStream:
    """Feeds bits (and infinite trailing zeros) to the decoder."""

    def __init__(self, bits):
        self.bits = bits
        self.index = 0

    def read(self):
        if self.index < len(self.bits):
            b = self.bits[self.index]
            self.index += 1
            return b
        else:
            return -1


def build_cumul(prob_vec: np.ndarray, total: int = 262144) -> np.ndarray:
    """
    Turn a length-N probability vector that sums to 1 into the length-(N+1)
    cumulative-frequency array expected by ArithmeticCoder. Ensures every
    symbol's frequency >= 1 and that the overall total == `total`.
    """
    alphabet_size = prob_vec.size

    freq = np.maximum(1, (prob_vec * (total - alphabet_size)).astype(np.int64))
    freq += 1
    diff = freq.sum() - total

    if diff > 0:
        for idx in np.argsort(-freq):
            take = min(freq[idx] - 1, diff)
            freq[idx] -= take
            diff -= take
            if diff == 0:
                break
    elif diff < 0:
        for idx in np.argsort(-prob_vec):
            freq[idx] += 1
            diff += 1
            if diff == 0:
                break

    assert freq.sum() == total, f"freq.sum={freq.sum()} != total={total}"
    assert np.all(freq >= 1)

    cumul = np.empty(alphabet_size + 1, dtype=np.int64)
    cumul[0] = 0
    cumul[1:] = np.cumsum(freq)
    return cumul


class LLMCompressor:
    """Wraps ArithmeticEncoder + cross-entropy tracking for AC-only compression."""

    def __init__(self):
        self.bitout = BitOutputStream()
        self.encoder = ArithmeticEncoder(32, self.bitout)
        self.cross_entropy_sum = 0.0
        self.token_count = 0

    def next_token(self, correct_token_idx, probs):
        prob = probs[correct_token_idx]
        if prob > 0:
            self.cross_entropy_sum += -math.log2(prob)
        else:
            self.cross_entropy_sum += float('inf')

        self.token_count += 1
        self.encoder.write(build_cumul(probs), correct_token_idx)

    def compress(self):
        self.encoder.finish()
        return self.bitout.get_bits()

    def get_cross_entropy(self):
        return self.cross_entropy_sum


class LLMDecompressor:
    def __init__(self, code):
        self.decoder = ArithmeticDecoder(32, BitInputStream(code))

    def decompress(self, probs) -> int:
        cumul = build_cumul(probs)
        return self.decoder.read(cumul, len(probs))


# =============================================================================
# Section 3: Tokenization + Model Inference (from prediction.py, AC-only)
# =============================================================================

class TokenDataPreparer:
    def __init__(self, args):
        if args.input_path is None and args.text_input is None:
            raise ValueError("Either input_path or text_input must be provided.")
        if args.input_path and args.text_input:
            raise ValueError("Only one of input_path or text_input can be provided.")

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")

        if args.input_path:
            self.data = self._get_data_from_file(args.input_path)
        else:
            self.data = args.text_input

        self.args = args

        print("Starting tokenization...")
        start_time = time.time()

        if args.first_n_tokens is not None:
            truncated_data = " ".join(self.data.split(" ", self.args.first_n_tokens)[:self.args.first_n_tokens])
            self.data_tokens = self.tokenizer.encode(truncated_data, truncation=True,
                                                       max_length=self.args.first_n_tokens)

            if len(self.data_tokens) < self.args.first_n_tokens:
                self.args.first_n_tokens = len(self.data_tokens)
                print(f"Reducing first_n_tokens to {self.args.first_n_tokens}, since the input data has fewer tokens.")
            assert len(self.data_tokens) == self.args.first_n_tokens, \
                f"Tokenization produced {len(self.data_tokens)} tokens, expected {self.args.first_n_tokens}."
        else:
            self.data_tokens = self.tokenizer.encode(self.data, truncation=False)
        print(f"Tokenization complete in {(time.time() - start_time):.2f}s. "
              f"Total number of tokens: {len(self.data_tokens)}")

        self.reduce_tokens = self.args.reduce_tokens

        if self.reduce_tokens:
            self.tokens_list = sorted(list(set(self.data_tokens)))
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        print(f"Total distinct tokens: {len(self.tokens_list)}")

    def _get_data_from_file(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data file not found: {input_path}")
        with open(input_path, 'r') as f:
            return f.read()

    def get_data_tokens(self):
        return self.data_tokens

    def get_bitmap(self):
        bitmap = BitMap(self.tokens_list)
        return bitmap.serialize()

    def get_args(self):
        return self.args


class TokenPredictor:
    """AC-only token predictor: always returns softmax probabilities."""

    def __init__(self, args, bitmap_data):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")
        self.args = args
        self.device = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS).")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, using CPU.")

        print(f"Loading model {args.model_name} on device {self.device}...")

        if args.engine == "transformer":
            if args.is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name, cache_dir=".cache", torch_dtype="auto")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.float16, cache_dir=".cache")

            print(f"Model {args.model_name} loaded with dtype {self.model.dtype}.")
            self.model.eval()
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported engine: {args.engine}")

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device=self.device)
        self.reduce_tokens = args.reduce_tokens

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one-step batched inference; returns (tokens_list, probs_cpu, data_copy_time, softmax_time).
        Always returns post-softmax probabilities (AC-only).
        """
        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        data_copy_time = 0
        with torch.inference_mode():
            if self.args.engine == "transformer":
                if not enable_kv_cache:
                    self._past_kv = None
                    self._cached_context_len = 0
                    t0_data_copy = time.perf_counter()
                    input_ids = torch.tensor(prompts, device=self.device)
                    data_copy_time += time.perf_counter() - t0_data_copy

                    if self.args.is_seq2seq:
                        if self.model.config.decoder_start_token_id is None:
                            self.model.config.decoder_start_token_id = 0
                        decoder_start = self.model.config.decoder_start_token_id
                        decoder_input_ids = torch.full(
                            (input_ids.shape[0], 1), decoder_start, dtype=torch.long, device=self.device)
                        outputs = self.model(
                            input_ids=input_ids, decoder_input_ids=decoder_input_ids, use_cache=False)
                    else:
                        outputs = self.model(input_ids, use_cache=enable_kv_cache)

                    self._past_kv = None
                    self._cached_context_len = input_ids.shape[1]
                else:
                    reset_cache = len(prompts[0]) < self._cached_context_len

                    if self._past_kv is None or reset_cache:
                        t0_data_copy = time.perf_counter()
                        input_ids = torch.tensor(prompts, device=self.device)
                        data_copy_time += time.perf_counter() - t0_data_copy
                        outputs = self.model(input_ids, use_cache=True)
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len = 0
                    else:
                        delta = [row[-1:] for row in prompts]
                        t0_data_copy = time.perf_counter()
                        delta = torch.tensor(delta, device=self.device, dtype=torch.long)
                        data_copy_time += time.perf_counter() - t0_data_copy
                        outputs = self.model(delta, past_key_values=self._past_kv, use_cache=True)
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len += 1
                logits = outputs.logits[:, -1, :]
            else:
                raise ValueError(f"Unsupported engine: {self.args.engine}")

            if getattr(self, "reduce_tokens", False):
                logits = logits.index_select(1, self.index_tensor.to(logits.device))

            t0_softmax = time.perf_counter()
            probs = torch.softmax(logits, dim=-1)
            softmax_time = time.perf_counter() - t0_softmax

            t0_data_copy = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0_data_copy

            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]


# =============================================================================
# Section 4: Compression / Decompression pipeline (from main.py, AC-only)
# =============================================================================

def run_global_mask_compression(args):
    """
    Run AC compression using a global token mask and batched LLM inference.
    Returns (first_tokens, bit_string, bitmask_data, stats, args).
    """
    print(f"\n----- Running Compression: Global Token Mask (tokens={args.first_n_tokens}, "
          f"kv_cache={args.use_kv_cache}) -----")

    t0_tokenize = time.perf_counter()
    input_token_cnt = 0

    token_data_preparer = TokenDataPreparer(args)
    data_tokens = token_data_preparer.get_data_tokens()
    args = token_data_preparer.get_args()
    if args.first_n_tokens is None:
        args.first_n_tokens = len(data_tokens)

    chunk_length = len(data_tokens) // args.batch_size
    extra = len(data_tokens) % args.batch_size

    batches = []
    start = 0
    for i in range(args.batch_size):
        size = chunk_length + (1 if i < extra else 0)
        end = start + size
        batches.append(data_tokens[start:end])
        start = end

    first_tokens = [batch[0] for batch in batches if batch]
    batches_length = [len(batch) for batch in batches]

    bitmask_data = token_data_preparer.get_bitmap()
    total_bitmap_size = len(bitmask_data) * 8
    tokenize_time = time.perf_counter() - t0_tokenize

    llm_compressor = LLMCompressor()
    token_predictor = TokenPredictor(args, bitmap_data=bitmask_data)

    prompts = [[] for _ in range(args.batch_size)]
    compression_time = time.perf_counter()
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0
    entropy = 0.0

    for token_idx in range(chunk_length):
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        for i in range(args.batch_size):
            prompts[i].append(batches[i][token_idx])

        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        input_token_cnt += args.batch_size * len(prompts[0])

        t0_inference = time.perf_counter()
        token_ids, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(
            prompts, args.use_kv_cache)
        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        actual_next_tokens = []
        valid_mask = []

        for idx in range(args.batch_size):
            if token_idx + 1 < batches_length[idx]:
                token = batches[idx][token_idx + 1]
                actual_next_tokens.append(token_ids.index(token))
                valid_mask.append(True)
            else:
                actual_next_tokens.append(0)
                valid_mask.append(False)

        t0_ac = time.perf_counter()
        probs_cpu = probs_values.to(torch.float16).numpy()

        for idx, probs in enumerate(probs_cpu):
            if not valid_mask[idx]:
                continue
            target_idx = actual_next_tokens[idx]
            llm_compressor.next_token(target_idx, probs)
            entropy += -np.log2(probs[target_idx])

        ac_time += time.perf_counter() - t0_ac

    bit_string = llm_compressor.compress()

    compression_time = time.perf_counter() - compression_time
    total_compression_time = time.perf_counter() - t0_tokenize

    total_arithmetic_code_size = len(bit_string)
    final_size = total_arithmetic_code_size + total_bitmap_size
    original_size_bytes = len(token_predictor.detokenize(data_tokens))

    return first_tokens, bit_string, bitmask_data, {
        "args": args.__dict__,
        "chunk_length": chunk_length,
        "original_size_bytes": original_size_bytes,
        "arithmetic_code_size_bytes": total_arithmetic_code_size / 8,
        "bitmap_size_bytes": total_bitmap_size / 8,
        "final_size_bytes": final_size / 8,
        "pure_compression_factor": original_size_bytes / (total_arithmetic_code_size / 8),
        "compression_factor": original_size_bytes / (final_size / 8),
        "input_tokens_count": input_token_cnt,
        "entropy": float(entropy),
        "total_compression_time": total_compression_time,
        "tokenize_time": tokenize_time,
        "compression_time": compression_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        "throughput_tokens_per_sec": input_token_cnt / total_compression_time,
        "throughput_kibibytes_per_sec": original_size_bytes / 1024 / total_compression_time,
        "inference_throughput_tokens_per_sec": input_token_cnt / inference_time,
        "inference_throughput_kibibytes_per_sec": original_size_bytes / 1024 / inference_time,
    }, args


def run_global_mask_decompression(args, first_tokens, bit_string, bitmap):
    """
    Decompress a bit string produced by run_global_mask_compression.
    Returns (reconstructed_tokens, detoken_string, stats).
    """
    print(f"\n----- Running Decompression: Global Token Mask (first_n_tokens={args.first_n_tokens}, "
          f"kv_cache={args.use_kv_cache}) -----")

    t0_decompress = time.perf_counter()

    token_predictor = TokenPredictor(args, bitmap_data=bitmap)

    decompressor = LLMDecompressor(bit_string)
    prompts = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size

    batches_length = [chunk_length + (1 if i < extra else 0) for i in range(args.batch_size)]
    input_tokens_cnt = 0
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0

    for token_idx in range(chunk_length):
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        input_tokens_cnt += args.batch_size * len(prompts[0])

        t0_inference = time.perf_counter()
        _, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(
            prompts, enable_kv_cache=args.use_kv_cache)
        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        t0_ac = time.perf_counter()
        for idx, probs in enumerate(probs_values.to(torch.float16).numpy()):
            if token_idx + 1 < batches_length[idx]:
                next_token_idx = decompressor.decompress(probs)
                next_token = token_predictor.get_token_by_id(next_token_idx)
                prompts[idx].append(next_token)
                reconstructed_tokens[idx].append(next_token)
        ac_time += time.perf_counter() - t0_ac

    reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))
    t0_detokenize = time.perf_counter()
    detoken_string = token_predictor.detokenize(reconstructed_tokens)
    detokenize_time = time.perf_counter() - t0_detokenize

    decompression_time = time.perf_counter() - t0_decompress

    return reconstructed_tokens, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / inference_time,
    }


# =============================================================================
# Section 5: File I/O utilities (from utils.py, trimmed)
# =============================================================================

def bits_to_bytes(bits):
    """Convert a list of bits (0/1 ints) into bytes. Returns (byte_data, padding_bits)."""
    bit_str = ''.join(str(b) for b in bits)
    padding = (8 - len(bit_str) % 8) % 8
    bit_str = bit_str + '0' * padding
    return int(bit_str, 2).to_bytes(len(bit_str) // 8, 'big'), padding


def bytes_to_bits(byte_data, padding):
    """Convert bytes back into a list of bits (0/1 ints), removing padding."""
    bit_str = bin(int.from_bytes(byte_data, 'big'))[2:].zfill(len(byte_data) * 8)
    bit_str = bit_str[:-padding] if padding > 0 else bit_str
    return [int(b) for b in bit_str]


def save_global_mask_file(args, first_token, bit_string, bitmask_data):
    """Save compression artifacts (header + first tokens + bitstream + bitmap) to a binary file."""
    file_path = args.output_path
    header = {
        "input_path": os.path.basename(args.input_path) if args.input_path else None,
        "model_name": args.model_name,
        "context_length": args.context_length,
        "first_n_tokens": args.first_n_tokens,
        "retain_tokens": args.retain_tokens,
        "use_kv_cache": args.use_kv_cache,
        "batch_size": args.batch_size,
    }
    with open(file_path, "wb") as f:
        header_str = json.dumps(header) + "\n"
        f.write(header_str.encode("utf-8"))

        for tok in first_token:
            f.write(struct.pack("I", tok))

        bit_bytes, padding = bits_to_bytes(bit_string)
        f.write(struct.pack("I", len(bit_bytes)))
        f.write(struct.pack("B", padding))
        f.write(bit_bytes)

        f.write(struct.pack("I", len(bitmask_data)))
        f.write(bitmask_data)


def load_global_mask_file(args):
    """Load compression artifacts from a binary file. Returns (args, first_token, bit_string, bitmask_data)."""
    file_path = args.input_path
    with open(file_path, "rb") as f:
        header_line = f.readline().decode("utf-8").strip()
        header = json.loads(header_line)

        first_token = [struct.unpack("I", f.read(4))[0] for _ in range(header["batch_size"])]

        bit_len = struct.unpack("I", f.read(4))[0]
        padding = struct.unpack("B", f.read(1))[0]
        bit_bytes = f.read(bit_len)
        bit_string = bytes_to_bits(bit_bytes, padding)

        bitmask_len = struct.unpack("I", f.read(4))[0]
        bitmask_data = f.read(bitmask_len)

    args.model_name = header["model_name"]
    args.context_length = header["context_length"]
    args.first_n_tokens = header["first_n_tokens"]
    args.retain_tokens = header["retain_tokens"]
    args.use_kv_cache = header["use_kv_cache"]
    args.batch_size = header["batch_size"]
    args.input_path = header["input_path"]

    return args, first_token, bit_string, bitmask_data


def load_results(results_file):
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}


def save_results(results, results_file):
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


def make_key(args):
    filename = os.path.basename(args.input_path) if args.input_path else "text_input"
    return (f"{filename}:{args.model_name}|ctx={args.context_length}|ret={args.retain_tokens}|"
            f"n={args.first_n_tokens}|kv={args.use_kv_cache}|batch={args.batch_size}|"
            f"reduce={args.reduce_tokens}|engine={args.engine}")


def create_run_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def save_params(args, run_dir):
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


# =============================================================================
# Section 6: Entry point
# =============================================================================

def main():
    args = get_main_args()

    if args.mode == "compress":
        first_tokens, bit_string, bitmask_data, stats, args = run_global_mask_compression(args)

        if args.print_results:
            print("\n\nCompression stats:")
            print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))

        if args.output_path:
            save_global_mask_file(args, first_tokens, bit_string, bitmask_data)
            print(f"\nSaved compressed file to: {args.output_path}")

    elif args.mode == "decompress":
        args, first_tokens, bit_string, bitmask_data = load_global_mask_file(args)
        reconstructed_tokens, detoken_string, stats = run_global_mask_decompression(
            args, first_tokens, bit_string, bitmask_data)

        if args.print_results:
            print("\n\nDecompression stats:")
            print(json.dumps({k: v for k, v in stats.items() if k != "args"}, indent=2))

        if args.output_path:
            with open(args.output_path, "w") as f:
                f.write(detoken_string)
            print(f"\nSaved reconstructed text to: {args.output_path}")
        else:
            print("\n\nReconstructed text:\n")
            print(detoken_string)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
