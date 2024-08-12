import os
import json
import torch
import argparse
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from vocab_count import count_freq, update_vocab_count_by_langfilter, count_recursive
from vocab_save import get_new_vocab_and_map, save_vocab
from model_save import saving_updated_qwenvl
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
    
    # valid check
    assert (args.support_data is not None) or (len(args.support_lang) > 0), "Must provide at least one pruning method." 

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
        
    
    # sub-token count
    print(f"==> Recursively calculate sub-token count")
    recur_counts = count_recursive(vocab_size=old_vocab_size, 
                                   vocab_counts=vocab_counts, 
                                   old_bytes_list=old_bytes_list)
    
    # get new vocab
    print(f"==> Get new vocabulary bpe file and save it")
    new_bytes_list, mapping_new2old = get_new_vocab_and_map(old_bytes_list=old_bytes_list, 
                                                            old_vocab_size=old_vocab_size,
                                                            vocab_counts=vocab_counts, 
                                                            recur_counts=recur_counts)
    new_vocab_size = len(mapping_new2old)
    save_vocab(new_bytes_list, mapping_new2old, args.new_model_path)

    # update model ckpt
    print(f"==> Update model ckpt for new tokenizer")
    saving_updated_qwenvl(old_model, new_vocab_size, mapping_new2old, args.new_model_path)
    
if __name__=='__main__':
    main()