import os
import json
import torch
import argparse
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from vocab_count import count_freq
from vocab_lang_filter import update_vocab_count_by_langfilter
from utils import get_bpe_file
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
    parser.add_argument('--support_lang', default=[], type=str, nargs='+')
    parser.add_argument('--inherit_vocab_count', type=str, default=None)
    args = parser.parse_args()
    
    # init output path
    if not os.path.exists(args.new_model_path):
        os.makedirs(args.new_model_path)
        print(f"==> Create output folder: {args.new_model_path}")
    
    # load old model and tokenizer
    print(f"==> Load old model and tokenizer from: {args.old_model_path}")
    old_model = AutoModelForCausalLM.from_pretrained(args.old_model_path, trust_remote_code=True)
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    old_vocab_size = old_model.config.__dict__['vocab_size']
    print(f"{old_model.config.__dict__['tokenizer_type']} has vocabulary size {old_vocab_size}")
    
    # using support data
    if args.support_data is not None:
        print(f"==> Loading Support Data (for Freqs Count) from: {args.support_data}")
        vocab_counts = count_freq(data_path=args.support_data, 
                                  vocab_size=old_vocab_size, 
                                  tokenizer=old_tokenizer, 
                                  output_path=args.new_model_path, 
                                  inherit_vocab_count=args.inherit_vocab_count)
    else:
        vocab_counts = [0 for _ in range(old_vocab_size)]
        
    # load bpe file
    tiktoken_bpe_file = get_bpe_file(args.old_model_path) 
    print(f"==> Load tiktoken bpe file from: {tiktoken_bpe_file}")
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    old_bytes_list = [base64.b64decode(token) for token, rank in (line.split() for line in contents.splitlines() if line)]
    
    # using supported language to filter
    if len(args.support_lang) > 0:
        print(f"==> Using support language to filter old vocabulary")
        print(f"Supported Language: {args.support_lang}")
        vocab_counts = update_vocab_count_by_langfilter(support_lang=args.support_lang, 
                                                        vocab_counts=vocab_counts, 
                                                        old_bytes_list=old_bytes_list, 
                                                        count_offset=1)
    
if __name__=='__main__':
    main()