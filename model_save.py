import os
import torch


def saving_updated_qwenvl(old_model, new_vocab_size, token_mapping, output_path):
    # define new module
    new_embeds = torch.nn.Embedding(new_vocab_size, old_model.config.hidden_size, dtype=old_model.transformer.wte.weight.dtype)
    new_lm_head = torch.nn.Linear(old_model.config.hidden_size, new_vocab_size, bias=False, dtype=old_model.lm_head.weight.dtype)
    # get new module parameter from the old
    assert len(set(token_mapping)) == new_vocab_size
    new_embeds.weight.data = old_model.transformer.wte.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    new_lm_head.weight.data = old_model.lm_head.weight.data[torch.LongTensor(token_mapping, device=old_model.device)]
    # update model
    old_model.transformer.wte.weight = new_embeds.weight
    old_model.lm_head.weight = new_lm_head.weight
    old_model.transformer.wte.num_embeddings = new_vocab_size
    old_model.lm_head.out_features = new_vocab_size
    # update config
    old_model.config.__dict__['vocab_size'] = new_vocab_size
    old_model.config.__dict__['_name_or_path'] = output_path
    old_model.config.__dict__['visual']["image_start_id"] = token_mapping.index(old_model.config.__dict__['visual']["image_start_id"])
    old_model.generation_config.__dict__['eos_token_id'] = token_mapping.index(old_model.generation_config.__dict__['eos_token_id'])
    old_model.generation_config.__dict__['pad_token_id'] = token_mapping.index(old_model.generation_config.__dict__['pad_token_id'])
    # save new model
    print(f"Saving new model ckpt to {output_path}")
    old_model.save_pretrained(output_path)