import os
from tqdm import tqdm
from langdetect import detect as langdetect
from langdetect import DetectorFactory
DetectorFactory.seed = 0 # no random

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
                
            