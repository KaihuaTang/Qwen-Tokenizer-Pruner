import torch
from tqdm import tqdm

def lossy_compression(targeted_vocab, token_counts, inherit_counts, old_token_list, original_vocab):
    total_count_with_idx = [(token_counts[i] + inherit_counts[i], i) for i in range(original_vocab)]
    sorted_count_with_idx = sorted(total_count_with_idx, key=lambda x: x[0])
    remove_count = 0
    remove_target = original_vocab - targeted_vocab
    removed_token = []
    for i in tqdm(range(len(sorted_count_with_idx))):
        token_count, token_idx = sorted_count_with_idx[i]
        if remove_count >= remove_target:
            continue
        elif token_count == 0:
            remove_count += 1
        elif len(old_token_list[token_idx]) > 1:
            # whether it can be represented by sub-token
            token = old_token_list[token_idx]
            b_len = len(token)
            for j in range(1, b_len):
                if (token[:j] in old_token_list) and (token[j:] in old_token_list):
                    parta_index = old_token_list.index(token[:j])
                    partb_index = old_token_list.index(token[j:])
                    if (token_counts[parta_index] + inherit_counts[parta_index] > 0) and (token_counts[partb_index] + inherit_counts[partb_index] > 0):
                        token_counts[token_idx] = 0
                        inherit_counts[token_idx] = 0
                        remove_count += 1
                        print("Removing: " + str(old_token_list[token_idx]))
                        break
    return token_counts, inherit_counts