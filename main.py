import os
import json
import torch
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils import make_context

from langdetect import detect as langdetect
from langdetect import DetectorFactory
# 确保检测结果的一致性
DetectorFactory.seed = 0

########## Prepare Dataset ###########
data_path = "/home/z00533370/projects/VLMEvalKit/raw_data/"

def get_text_list(folder_path):
    query_list = []
    response_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('json'):
            file = json.load(open(os.path.join(folder_path, file_name)))
            query_list.append(file['query'])
            response_list.append(file['response'])
    return query_list, response_list

query_list, response_list = get_text_list(data_path)

########## Count Token Freqs ###########
model_path = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
token_counts = [0 for _ in range(151936)]
assert len(query_list) == len(response_list)
for i in tqdm(range(len(query_list))):
    query, response = query_list[i], response_list[i]
    _, context_tokens = make_context(tokenizer, query, history=[], system="You are a helpful assistant.")
    for token in context_tokens:
        token_counts[token] += 1
    response_tokens = tokenizer.encode(response)
    for token in response_tokens:
        token_counts[token] += 1

########## Dictionary Pruning ###########
tiktoken_bpe_file = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197/qwen.tiktoken"

def is_special_token(token):
    return ((token.startswith('<') and token.endswith('>') and len(token) > 2) or
            (token.startswith('[') and token.endswith(']') and len(token) > 2))

with open(tiktoken_bpe_file, "rb") as f:
    contents = f.read()
old_token_list = [base64.b64decode(token) for token, rank in (line.split() for line in contents.splitlines() if line)]
new_token_list = []
mapping_new2old = []
# detect language, only keep english and chinese
for i in tqdm(range(len(old_token_list))):
    token = old_token_list[i]
    try:
        # number and symbols cannot be detected by langdetect
        token_str = token.decode("utf-8")
        if (langdetect(token_str) in ['zh-cn', 'en']) or (token_counts[i] > 0) or is_special_token(token_str):
            new_token_list.append(token)
            mapping_new2old.append(i)
    except:
        new_token_list.append(token)
        mapping_new2old.append(i)

########## Add Special Token Mapping ###########
qwen_vocab_size = 151936
for i in range(len(old_token_list), qwen_vocab_size):
    mapping_new2old.append(i)

########## Save New Vocab ###########
new_tiktoken_bpe_file = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197-new-vocab/qwen.tiktoken"
vocab_mapping_file = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197-new-vocab/token_vocab_mapping.torch"
# saving new tiktoken_bpe_file
with open(new_tiktoken_bpe_file, "w", encoding="utf8") as w:
    for i, token in enumerate(new_token_list):
        line = base64.b64encode(token).decode("utf8") + " " + str(i) + "\n"
        w.write(line)
# saving mapping index
torch.save(torch.LongTensor(mapping_new2old), vocab_mapping_file)

########## update model ###########
old_model_path = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197/"
new_model_path = "/home/z00533370/projects/MoH/exp0731_qwenvl_chat_moh_layer16-31_sigmoid_prob_no_norm/checkpoint-5197-new-vocab/"
model = AutoModelForCausalLM.from_pretrained(old_model_path, trust_remote_code=True)
# define new module
new_embeds = torch.nn.Embedding(len(mapping_new2old), model.config.hidden_size, dtype=model.dtype)
new_lm_head = torch.nn.Linear(model.config.hidden_size, len(mapping_new2old), bias=False, dtype=model.dtype)
# get new module parameter from the old
assert len(set(mapping_new2old)) == len(mapping_new2old)
new_embeds.weight.data = model.transformer.wte.weight.data[torch.LongTensor(mapping_new2old, device=model.device)]
new_lm_head.weight.data = model.lm_head.weight.data[torch.LongTensor(mapping_new2old, device=model.device)]
# update model
model.transformer.wte.weight = new_embeds.weight
model.lm_head.weight = new_lm_head.weight
model.transformer.wte.num_embeddings = len(mapping_new2old)
model.lm_head.out_features = len(mapping_new2old)
# update config
model.config.__dict__['vocab_size'] = len(mapping_new2old)
model.config.__dict__['_name_or_path'] = new_model_path
model.generation_config.__dict__['eos_token_id'] = mapping_new2old.index(model.generation_config.__dict__['eos_token_id'])
model.generation_config.__dict__['pad_token_id'] = mapping_new2old.index(model.generation_config.__dict__['pad_token_id'])
# save new model
model.save_pretrained(new_model_path)
