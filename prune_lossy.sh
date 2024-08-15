#!/bin/sh

old_model_path="../../checkpoints/Qwen-7B-Chat/"
new_model_path="../../checkpoints/Qwen-7B-Chat-new-vocab/"
support_data="../../VLMEvalKit/raw_data/"
support_lang="zh-cn en" # optional   e.g., support_lang="zh-cn en"
inherit_vocab_count="" # optional
target_size=30000

# run pruning, check whether optional args are exists
if [ -z "$support_lang" -a -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --target_size $target_size"
elif [ -z "$inherit_vocab_count" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --support_lang $support_lang --target_size $target_size"
elif [ -z "$support_lang" ]; then
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --inherit_vocab_count $inherit_vocab_count --target_size $target_size"
else
    cmd="python main.py --old_model_path $old_model_path --new_model_path $new_model_path --support_data $support_data --support_lang $support_lang --inherit_vocab_count $inherit_vocab_count --target_size $target_size"
fi
echo $cmd
$cmd