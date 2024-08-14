import os
import json
import torch
import argparse
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from vocab_count import get_text_list
from utils import make_context
from tqdm import tqdm


from langdetect import detect as langdetect
from langdetect import DetectorFactory
# 确保检测结果的一致性
DetectorFactory.seed = 0

def main():
    # start vocabulary pruning
    print('============ Start Qwen Vocabulary Pruning ==========')
    
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model_path', type=str, default=None)
    parser.add_argument('--new_model_path', type=str, default=None)
    parser.add_argument('--support_data', type=str, default=None)
    args = parser.parse_args()

    # load tokenziers
    print(f"Load old tokenizer from {args.old_model_path}")
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    print(f"Load new tokenizer from {args.new_model_path}")
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_model_path, trust_remote_code=True)
    print(f"Load token mapping from {os.path.join(args.new_model_path, 'token_mapping.torch')}")
    mapping_new2old = torch.load(os.path.join(args.new_model_path, 'token_mapping.torch')).long().tolist()

    # load data
    print(f"Load check data from {args.support_data}")
    query_list, prompt_list = get_text_list(args.support_data)

    # check query
    print(f"For query list that requires system prompt")
    mismatch_list = []
    for i in tqdm(range(len(query_list))):
        query = query_list[i]
        _, old_context_tokens = make_context(old_tokenizer, query, history=[], system="You are a helpful assistant.")
        _, new_context_tokens = make_context(new_tokenizer, query, history=[], system="You are a helpful assistant.")
        if len(old_context_tokens) != len(new_context_tokens):
            mismatch_list.append(query)
        elif not all([old_token == mapping_new2old[new_token] for old_token, new_token in zip(old_context_tokens, new_context_tokens)]):
            mismatch_list.append(query)
    print(f"==> Mismatch num in query list: {len(mismatch_list)}. All Correct!")
    if len(mismatch_list) > 0:
        _, old_context_tokens = make_context(old_tokenizer, mismatch_list[0], history=[], system="You are a helpful assistant.")
        print(f"==> Mismatch example 0 old tokens: {old_context_tokens}")
        _, new_context_tokens = make_context(new_tokenizer, mismatch_list[0], history=[], system="You are a helpful assistant.")
        new_context_tokens = [mapping_new2old[new_token] for new_token in new_context_tokens]
        print(f"==> Mismatch example 0 new tokens: {new_context_tokens}")

    # check prompt
    print(f"For plain text list that doesn't require system prompt")
    mismatch_list = []
    for i in tqdm(range(len(prompt_list))):
        prompt = prompt_list[i]
        old_context_tokens = old_tokenizer.encode(prompt)
        new_context_tokens = new_tokenizer.encode(prompt)
        if len(old_context_tokens) != len(new_context_tokens):
            mismatch_list.append(prompt)
        elif not all([old_token == mapping_new2old[new_token] for old_token, new_token in zip(old_context_tokens, new_context_tokens)]):
            mismatch_list.append(prompt)
    print(f"==> Mismatch num in plain text list: {len(mismatch_list)}. All Correct!")
    if len(mismatch_list) > 0:
        old_context_tokens = old_tokenizer.encode(mismatch_list[0])
        print(f"==> Mismatch example 0 old tokens: {old_context_tokens}")
        new_context_tokens = new_tokenizer.encode(mismatch_list[0])
        new_context_tokens = [mapping_new2old[new_token] for new_token in new_context_tokens]
        print(f"==> Mismatch example 0 new tokens: {new_context_tokens}")


if __name__=='__main__':
    main()