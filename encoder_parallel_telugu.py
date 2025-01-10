import time
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import re

# Function to encode a chunk of tokens into UTF-8 and return as bytes
def encode_chunk(chunk):
    # Encode each token in the chunk to UTF-8
    return [token.encode('utf-8') for token in chunk]

# Main function to handle parallel encoding and return concatenated results
def encode_tokens_parallel(tokens, chunk_size=1_000_000, max_workers=10):
    # Split the tokens into chunks of size chunk_size (1 million tokens per chunk)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # Prepare the progress bar
    total_chunks = len(chunks)
    
    # Use ProcessPoolExecutor to process chunks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show a progress bar while processing chunks
        encoded_chunks = list(tqdm(executor.map(encode_chunk, chunks), total=total_chunks, desc="Processing Chunks"))
    
    # Concatenate all encoded chunks into a single list
    concatenated_encoded = [token for chunk in encoded_chunks for token in chunk]
    
    return concatenated_encoded

def load_telugu_texts():
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
    # Concatenate all texts and remove all English, numerical values, quotes, and characters outside the UTF-8 range 0x0C00 to 0x0C7F, including special characters like @, #, $, and %.
    telugu_text = ' '.join(telugu_texts)
    telugu_text = re.sub(r'[^\u0C00-\u0C7F@#$%]', '', telugu_text)  # Remove characters outside the specified UTF-8 range and special characters
    telugu_text = re.sub(r'[\r\n\xa0]', '', telugu_text)  # Remove line breaks and non-breaking spaces
    return telugu_text

# Main script
if __name__ == '__main__':
    # Load the Telugu texts
    tokens = load_telugu_texts()
    # Start the timer
    start_time = time.time()

    # Encode the tokens in parallel and get concatenated results
    encoded_tokens = encode_tokens_parallel(tokens, chunk_size=1_000_000, max_workers=10)
    print(encoded_tokens[:100])
    print(len(encoded_tokens))
    # End the timer
    end_time = time.time()

    # Calculate the time taken
    time_taken = end_time - start_time

    print(f"Time taken to encode and process tokens in parallel: {time_taken:.4f} seconds")
    print("Encoding and processing completed!")
