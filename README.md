# Qwen-Tokenizer-Pruner
Due to the huge vocaburary size (151,936) of Qwen models, the Embedding and LM Head weights are excessively heavy. Therefore, this project provides a Tokenizer vocabulary shearing solution for Qwen and Qwen-VL.

**If my open source projects have inspired you, giving me some sponsorship will be a great help to my subsequent open source work.** 
[Support my subsequent open source work❤️🙏](https://kaihuatang.github.io/donate.html) [(Previous Supporters)](https://kaihuatang.github.io/supporters.html)

## Installation
Run the following command to install required packages
```
pip install -r requirements.txt
```

## Supported Models
This tokenizer vocabulary pruning tool supports the following LLM models.
- [Qwen]()
- [Qwen-VL]

Please download your base model from the above checkpoints.

## Getting Started

1. Get new model and tokenzer with smaller vocabulary size
```
python main.py --old_model_path ~/projects/checkpoints/Qwen-1_8B-Chat/ --new_model_path ~/projects/checkpoints/Qwen-1_8B-Chat-New-Vocab/ --support_data ./sample_data --support_lang 'zh-cn' 'en'
```

2. Check whether the new tokenizer is equal to the original tokenizer
```
python check.py --old_model_path ~/projects/checkpoints/Qwen-1_8B-Chat/ --new_model_path ~/projects/checkpoints/Qwen-1_8B-Chat-New-Vocab/ --support_data ./sample_data
```

## Prepare Your Own Target Dataset

## License and Citation