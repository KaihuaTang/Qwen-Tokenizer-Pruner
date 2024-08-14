import os
from tqdm import tqdm
import json
import torch
from utils import make_context
from langdetect import detect as langdetect
from langdetect import DetectorFactory
DetectorFactory.seed = 0 # no random

def get_text_list(folder_path):
    query_list = []
    prompt_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('json'):
            file = json.load(open(os.path.join(folder_path, file_name)))
            if 'query' in file:
                query_list.append(file['query'])
            if 'response' in file:
                prompt_list.append(file['response'])
            if 'prompt' in file:
                prompt_list.append(file['prompt'])
    return query_list, prompt_list


def count_freq(data_path, vocab_size, tokenizer, output_path, inherit_vocab_count):
    vocab_counts = [0 for _ in range(vocab_size)]
    # load data
    query_list, prompt_list = get_text_list(data_path)
    # calculate query vocabs
    print("calculate query vocab counts: add system prompt before encode")
    for i in tqdm(range(len(query_list))):
        query = query_list[i]
        _, context_tokens = make_context(tokenizer, query, history=[], system="You are a helpful assistant.")
        for token in context_tokens:
            vocab_counts[token] += 1
     
    # calculate promopt vocabs        
    print("calculate prompt vocab counts: encode directly")
    for i in tqdm(range(len(prompt_list))):
        prompt = prompt_list[i]
        prompt_tokens = tokenizer.encode(prompt)
        for token in prompt_tokens:
            vocab_counts[token] += 1
            
    # add inherit vocab if it's not none
    if (inherit_vocab_count is not None):
        if os.path.exists(inherit_vocab_count):
            print(f"==> Load inherit_vocab_count and add it to current vocab_counts: path({inherit_vocab_count})")
            inherit_vocab_count = torch.load(inherit_vocab_count)
            assert len(inherit_vocab_count) == vocab_size, f"inherit_vocab_count (size: {len(inherit_vocab_count)}) should have the same vocab size {vocab_size}"
            for token, i_count in enumerate(inherit_vocab_count):
                vocab_counts[token] += int(i_count)
        else:
            print(f"==> No valid inherit vocabulary count path, skip inheritance!")
    
    # save vocab_counts
    torch.save(vocab_counts, os.path.join(output_path, 'vocab_counts.torch'))
    return vocab_counts


def is_special_token(token):
    return ((token.startswith('<') and token.endswith('>') and len(token) > 2) or
            (token.startswith('[') and token.endswith(']') and len(token) > 2))
    

def update_vocab_count_by_langfilter(support_lang, vocab_counts, old_bytes_list, count_offset=1):
    for i in tqdm(range(len(old_bytes_list))):
        token_bytes = old_bytes_list[i]
        # add try to keep unknown token 
        try:
            token_str = token_bytes.decode("utf-8")
            if (langdetect(token_str) in support_lang) or is_special_token(token_str):
                vocab_counts[i] += count_offset
        except:
            vocab_counts[i] += count_offset
    return vocab_counts
                

def count_recursive(vocab_size, vocab_counts, old_bytes_list):
    recursive_counts = [0 for _ in range(vocab_size)]

    for i in tqdm(range(len(old_bytes_list))):
        token_bytes = old_bytes_list[i]
        t_count = vocab_counts[i]
        b_len = len(token_bytes)
        if t_count > 0 and b_len > 1:
            for j in range(1, b_len):
                for k in range(b_len+1-j):
                    sub_token = token_bytes[k:j+k]
                    if sub_token in old_bytes_list:
                        recursive_counts[old_bytes_list.index(sub_token)] += t_count

    return recursive_counts