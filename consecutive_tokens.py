from collections import OrderedDict

def get_consecutive_tokens(li, window_size=4):
    if len(li) == 0:
        return []
    final_token_dict = OrderedDict((token, []) for token in range(len(li)))
    i = 0
    while i <= len(li)-1:
        j = 1
        while j <= window_size:
            final_token_dict[i].append(tuple(li[i:i+j]))
            j+=1
        i+=1
    
    reversed_token_dict = {key: [tuple(tup) for tup in reversed(value)] for key, value in final_token_dict.items()}
    return reversed_token_dict



def search_consecutive_tokens(ordered_dict, encoded_token_dict):
    final_encoded_tokens = []
    keys = list(ordered_dict.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        j = 0
        jump = False
        while j<len(ordered_dict[key]):
            if ordered_dict[key][j] in encoded_token_dict:
                final_encoded_tokens.append(encoded_token_dict[ordered_dict[key][j]])
                i+=len(ordered_dict[key][j])
                jump = True
                j = 0
                break
            j+=1
        if not jump:
            i+=1
    return final_encoded_tokens

if __name__ == "__main__":
    text = "తెలుగు భాష ఒక ద్రావిడ భాష."
    op_li = get_consecutive_tokens([1,2,3,4,5])
    print(op_li)

    dict = {(1,2):9,(3,):10, (4,5):11}
    opp = search_consecutive_tokens(op_li, dict)
    print(opp)
    text = "9,10,11"
    toks_li = [token for token in text.split(',')]
    # Reading vocabulary from file
    import tokenizer
    formatted_vocab = tokenizer.read_vocab_from_file()
    # Invert vocabulary
    inverted_vocab = {v: k for k, v in formatted_vocab.items()}
    # Expand vocabulary
    decoder_map = tokenizer.expand_vocab(inverted_vocab)
    decoded_tokens = [decoder_map.get(int(token)) for token in toks_li]
    print(decoded_tokens)
    # encoded_tokens = encode_tokens_parallel(text, chunk_size=1_000_000, max_workers=2)
    # encoded_tokens = [token.encode('utf-8') for token in text]
    # decoded_tokens = [i.decode('utf-8') for i in encoded_tokens]
    # print(get_consecutive_tokens(decoded_tokens))
