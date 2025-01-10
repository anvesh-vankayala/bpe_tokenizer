import pandas as pd
import re
import encoder_parallel_telugu as encode_parallel
import time
import json
from collections import defaultdict
from tqdm import tqdm

def load_and_encode_tokens():
    tokens = encode_parallel.load_telugu_texts()
    start_time = time.time()
    encoded_tokens = encode_parallel.encode_tokens_parallel(tokens, chunk_size=1_000_000, max_workers=10)
    print('encoded_tokens:', encoded_tokens[:100])
    print(len(encoded_tokens))
    end_time = time.time()
    print(f"Time taken to encode and process tokens in parallel: {end_time - start_time:.4f} seconds")
    print('length of encoded_text:', len(encoded_tokens))
    print('unique characters in decoded_text:', {token.decode('utf-8') for token in set(encoded_tokens)})
    # print('unique characters in encoded_text:', set(encoded_tokens))
    print('unique characters in encoded_text:', len(set(encoded_tokens)))
    return encoded_tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def bpe_process(encoded_tokens,vocab_size=500, encoded_tokens_length=10_00_000):
    num_merges = vocab_size - 256  # our unique tokens are 194, for our sample text.
    encoded_tokens = encoded_tokens[:encoded_tokens_length]
    ids = list(encoded_tokens)  # copy so we don't destroy the original list
    merges = {}  # (int, int) -> int

    for i in tqdm(range(num_merges), desc="Merging tokens"):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx  # merge has a pair of tokens and the new token index

    print("tokens length:", len(encoded_tokens))
    print("ids length:", len(ids))
    print("by paired tokens length:", len(set(ids)))
    print(f"compression ratio: {len(encoded_tokens) / len(ids):.2f}X")
    # print(f"token size: {len(set(encoded_tokens))}")

    return merges

def build_vocabulary(merges):
    telugu_unicode_chars = [chr(i) for i in range(0x0C00, 0x0C7F)]  # Telugu Unicode range
    vocab = {token: idx for token, idx in merges.items()}

    for idx, char in enumerate([chr(i).encode('utf-8') for i in range(0x0C00, 0x0C7F)]):
        if idx < 256:  # Ensure we only add up to 256 characters
            vocab[char] = idx  # Map the character to its index

    vocab[b' '] = 255
    vocab[b'.'] = 254

    with open('merges_vocab.json', 'w') as f:
        json.dump({'merges': {str(k): v for k, v in merges.items()}, 'vocab': {str(k): v for k, v in vocab.items()}}, f)

def read_vocab_from_file():
    with open('merges_vocab.json', 'r') as f:
        data = json.load(f)

    distributed_data = defaultdict(list)

    for key, value in data['vocab'].items():
        distributed_data['vocab'].append({key: value})

    formatted_vocab = {}
    for item in distributed_data['vocab']:
        for k, v in item.items():
            if ',' not in k:
                formatted_vocab[(eval(k),)] = v
            else:
                formatted_vocab[eval(k)] = v

    return formatted_vocab

def expand_vocab(inverted_vocab):
    def convert_to_bytes(value):
        if isinstance(value, bytes):
            return value
        elif value in inverted_vocab:
            return process_tuple(inverted_vocab[value])
        else:
            print(f'value not found in inverted_vocab: {value}')
            return None

    def process_tuple(value_tuple):
        converted_values = []
        for v in value_tuple:
            result = convert_to_bytes(v)
            if isinstance(result, tuple):
                converted_values.extend(result)
            else:
                converted_values.append(result)
        return tuple(converted_values)

    decoder_map = {k: process_tuple(v) for k, v in inverted_vocab.items()}
    print("sample decoder map:", {k: decoder_map[k] for k in list(decoder_map)[:5]})
    return decoder_map

# Main execution
if __name__ == "__main__":
    # 1. Load and encode tokens
    encoded_tokens = load_and_encode_tokens()
    # 2. Process BPE
    merges = bpe_process(encoded_tokens,vocab_size=5000, encoded_tokens_length=50_00_000)
    # 3. Build vocabulary
    build_vocabulary(merges)
    # 4. Read vocabulary from file
    formatted_vocab = read_vocab_from_file()
    # 5. Invert vocabulary
    inverted_vocab = {v: k for k, v in formatted_vocab.items()}
    # 6. Expand vocabulary
    decoder_map = expand_vocab(inverted_vocab)
    # 7. Invert back again
    re_inverted_vocab = {k: v for v, k in decoder_map.items()}
    # print(re_inverted_vocab)
