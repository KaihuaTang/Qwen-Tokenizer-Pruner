import os
import torch
import base64
from tqdm import tqdm

def get_new_vocab_and_map(old_bytes_list, old_vocab_size, vocab_counts, recur_counts):
    new_bytes_list = []
    mapping_new2old = []

    for i in tqdm(range(len(old_bytes_list))):
        if vocab_counts[i] + recur_counts[i] > 0:
            new_bytes_list.append(old_bytes_list[i])
            mapping_new2old.append(i)

    # Add Special Token Mapping 
    print(f"Add special token (num: {old_vocab_size - len(old_bytes_list)})")
    for i in range(len(old_bytes_list), old_vocab_size):
        mapping_new2old.append(i)

    print(f"Vocaburaly size: {old_vocab_size} => {len(mapping_new2old)}")

    return new_bytes_list, mapping_new2old

def save_vocab(bytes_list, token_mapping, output_path):
    new_tiktoken_path = os.path.join(output_path, 'qwen.tiktoken')
    token_mapping_path = os.path.join(output_path, 'token_mapping.torch')
    # saving new tiktoken_bpe_file
    with open(new_tiktoken_path, "w", encoding="utf8") as w:
        for i, token in enumerate(bytes_list):
            line = base64.b64encode(token).decode("utf8") + " " + str(i) + "\n"
            w.write(line)
    print(f"New Tiktoken BPE file (size: {len(bytes_list)}) is saved to {new_tiktoken_path}")

    # saving mapping index
    torch.save(torch.LongTensor(token_mapping), token_mapping_path)
    print(f"Mapping file (new token 2 old token) is saved: {token_mapping_path}")


