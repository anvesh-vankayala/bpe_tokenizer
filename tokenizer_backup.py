import pandas as pd
import re

# Data section  start--> 
# Load the CSV files
file_paths = [
    '/Users/anvesh/codebase/llm/data/telugu_books/telugu_books.csv',
    '/Users/anvesh/codebase/llm/data/telugu_news/1_telugu_news.csv',
    '/Users/anvesh/codebase/llm/data/telugu_news/2_telugu_news.csv'
]

# Combine data from all files
telugu_texts = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    if 'text' in df.columns:
        telugu_texts.append(' '.join(df['text'].astype(str).tolist()))
    elif 'body' in df.columns:
        telugu_texts.append(' '.join(df['body'].astype(str).tolist()))

# Concatenate all texts and remove all English, numerical values, and quotes
telugu_text = ' '.join(telugu_texts)
telugu_text = re.sub(r'[A-Za-z0-9\'"]', '', telugu_text)  # Remove English letters, numbers, and quotes
telugu_text = re.sub(r'[\r\n\xa0]', '', telugu_text)  # Remove line breaks and non-breaking spaces

print('telugu_text befores utf-8 encoding:', telugu_text[:100])

vocabulary_size = len(set(telugu_text.split()))
print('Original text size:', len(telugu_text))
print('Vocabulary size of telugu_text:', vocabulary_size)

unique_characters = set(telugu_text)
unique_count = len(unique_characters)
print('Original text size:', len(telugu_text))
print('Unique character count in telugu_text:', unique_count)

# Data section  end--> 

# utf-8 encoding section start -->
import encode_parallel_telugu as encode_parallel
import time

tokens = encode_parallel.load_telugu_texts()
# Start the timer
start_time = time.time()
# Encode the tokens in parallel and get concatenated results
encoded_tokens = encode_parallel.encode_tokens_parallel(tokens, chunk_size=1_000_000, max_workers=10)
print('encoded_tokens:', encoded_tokens[:100])
print(len(encoded_tokens))
# End the timer
end_time = time.time()
print(f"Time taken to encode and process tokens in parallel: {end_time - start_time:.4f} seconds")

print('length of encoded_text:', len(encoded_tokens))
print('unique characters in encoded_text:', set(encoded_tokens))
print('unique characters in encoded_text:', len(set(encoded_tokens)))
# utf-8 encoding section end -->

# BPE section start -->
#### **BPE implementation**

tokens = encoded_tokens

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# ---
vocab_size = 500 # the desired final vocabulary size
num_merges = vocab_size - 256 ## our unique tokens are 194, for our sample text.
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
from tqdm import tqdm  # Import tqdm for progress bar

for i in tqdm(range(num_merges), desc="Merging tokens"):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    # print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx # merge has a pair of tokens and the new token index
    
print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
print(f"token size: {len(set(tokens))}")
    
# print(ids)
# BPE section end -->

# Building the vocabulary section start -->
telugu_unicode_chars = [chr(i) for i in range(0x0C00, 0x0C7F)]  # Telugu Unicode range

# Add these characters to the vocabulary
import json
vocab = {token: idx for token, idx in merges.items()}
# Add unique Telugu characters to the vocabulary
for idx, char in enumerate([chr(i).encode('utf-8') for i in range(0x0C00, 0x0C7F)]):
    if idx < 256:  # Ensure we only add up to 256 characters
        vocab[char] = idx  # Map the character to its index

vocab[b' '] = 255
vocab[b'.'] = 254
# Save merges and vocab to a file
# with open('merges_vocab.json', 'w') as f:
#     json.dump({'merges': merges, 'vocab': vocab}, f)

# saving the merges and vocab to a file
with open('merges_vocab.json', 'w') as f:
    json.dump({'merges': {str(k): v for k, v in merges.items()}, 'vocab': {str(k): v for k, v in vocab.items()}}, f)
    
# Building the vocabulary section end -->


# Reading the merges and vocab from a file section start -->
import json
from collections import defaultdict

# Read the merges and vocab data from the JSON file
with open('merges_vocab.json', 'r') as f:
    data = json.load(f)

# Create a defaultdict to store the data in a distributed manner
distributed_data = defaultdict(list)

# Distribute the merges and vocab data
# for key, value in data['merges'].items():
#     distributed_data['merges'].append({key: value})

for key, value in data['vocab'].items():
    distributed_data['vocab'].append({key: value})

# Optionally, print the distributed data for verification
print(distributed_data)
distributed_data['vocab']
# Convert the list of dictionaries to a single dictionary
formatted_vocab = {}
for item in distributed_data['vocab']:
    for k, v in item.items():
        if ',' not in k:
            formatted_vocab[(eval(k),)] = v
        else:
            formatted_vocab[eval(k)] = v
print(formatted_vocab[:50])
# inverting the vocab
inverted_vocab = {v: k for k, v in formatted_vocab.items()}
inverted_vocab

# Reading the merges and vocab from a file section end -->

# Expanding the vocab section start -->
def convert_to_bytes(value):
    if isinstance(value, bytes):
        return value
    elif value in inverted_vocab:
        return process_tuple(inverted_vocab[value])
    else:
        print(f'value not found in inverted_vocab: {value}')
        return None

def process_tuple(value_tuple):
    # print(f'value_tuple: {value_tuple}')
    # for vi in value_tuple:
    #     print(f'v: {vi}')
    converted_values = []
    for v in value_tuple:
        result = convert_to_bytes(v)
        if isinstance(result, tuple):
            converted_values.extend(result)
        else:
            converted_values.append(result)
    return tuple(converted_values)

decoder_map = {k: process_tuple(v) for k, v in inverted_vocab.items()}





