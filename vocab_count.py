import os
from tqdm import tqdm
import json
import torch
from utils import make_context

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
    print("calculate query vocabs: add system prompt before encode")
    for i in tqdm(range(len(query_list))):
        query = query_list[i]
        _, context_tokens = make_context(tokenizer, query, history=[], system="You are a helpful assistant.")
        for token in context_tokens:
            vocab_counts[token] += 1
     
    # calculate promopt vocabs        
    print("calculate prompt vocabs: encode directly")
    for i in tqdm(range(len(prompt_list))):
        prompt = prompt_list[i]
        prompt_tokens = tokenizer.encode(prompt)
        for token in prompt_tokens:
            vocab_counts[token] += 1
            
    # add inherit vocab if it's not none
    if inherit_vocab_count is not None:
        print(f"==> Load inherit_vocab_count and add it to current vocab_counts: path({inherit_vocab_count})")
        inherit_vocab_count = torch.load(inherit_vocab_count)
        assert len(inherit_vocab_count) == vocab_size, f"inherit_vocab_count (size: {len(inherit_vocab_count)}) should have the same vocab size {vocab_size}"
        for token, i_count in enumerate(inherit_vocab_count):
            vocab_counts[token] += int(i_count)
    
    # save vocab_counts
    torch.save(vocab_counts, os.path.join(output_path, 'vocab_counts.torch'))
    return vocab_counts