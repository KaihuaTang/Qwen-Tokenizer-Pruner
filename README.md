<p align="left">
    <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp
</p>


# Qwen-Tokenizer-Pruner
Due to the huge vocaburary size (151,936) of Qwen models, the Embedding and LM Head weights are excessively heavy. Therefore, this project provides a Tokenizer vocabulary pruning solution for Qwen and Qwen-VL.

**If my open source projects have inspired you, giving me some sponsorship will be a great help to my subsequent open source work.** 
[Support my subsequent open source work❤️🙏](https://kaihuatang.github.io/donate.html) [(Previous Supporters)](https://kaihuatang.github.io/supporters.html)


## Contents
1. [Installation](#installation)
2. [Supported Models](#supported-models)
3. [Getting Started](#getting-started)
    - [1. Lossless Pruning](#1-lossless-pruning)
    - [2. Lossy Pruning](#2-lossy-pruning)
    - [3. Other Details and Special Cases](#3-other-details-and-special-cases)
4. [Prepare Your Own Support Dataset](#prepare-your-own-support-dataset)
5. [Citation](#citation)


## Installation
Run the following command to install required packages
```
pip install -r requirements.txt
```

## Supported Models
This tokenizer vocabulary pruning tool supports the following LLM models.
- [Qwen-series](https://huggingface.co/collections/Qwen/qwen-65c0e50c3f1ab89cb8704144)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)
- [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)

Please download your base model from the above checkpoints.

## Getting Started
We support two types of tokenizer vocabulary pruning: lossless (in support data) and lossy (to a target size)

### 1. Lossless Pruning
To conduct lossless vocabulary pruning, you just need to simply run the following script with your own data/model pathes.
```
bash prune_lossless.sh
```
The script will first prune the vocabulary and save it to the output path, and then check whether old tokenizer and new tokenzer are equivalent.

Explaination of arguments used in the script
```
old_model_path="../../checkpoints/Qwen-VL-Chat/"
new_model_path="../../checkpoints/Qwen-VL-Chat-new-vocab/"
support_data="../../VLMEvalKit/raw_data/"
support_lang="" # optional (using "langdetect")   e.g., support_lang="zh-cn en"
inherit_vocab_count="" # optional
```

### 2. Lossy Pruning
Run the following bash script can conduct lossy vocabulary pruning to a target size.
```
bash prune_lossy.sh
```
This script add an argument 'target_size', which will remove the less frequent token and cause mismatch between old tokenizer and new tokenizer. Therefore, it will no longer conduct equivalence check.


### 3. Other details and special cases:
- For support_lang, note that language detection is using [langdetect](https://pypi.org/project/langdetect/) package, please using the valid abbreviations of languages.
- Post processing
For Qwen models, change SPECIAL_START_ID in tokenization_qwen.py to your New Tiktoken BPE file Size, check printed log (see the following example). 
![alt text](./assets/example.png "New SPECIAL_START_ID")


## Prepare Your Own Support Dataset
We provide a list of sample data in "./sample_data/x.json" as an example of support data used in vocabulary pruning. Each file is either a dictionary of query and response, or a dictionary for a plain text.

Support Data Format A:
```
{
    "query": Picture 1: <img>/YOUR_OWN_PATH/MMBench/demo.jpg</img>\nWhat is in the image? (This query will be tokenized with system prompt)",
    "response": "A white cat. (This response will be directly tokenized from plain text)"
}
```

Support Data Format B:
```
{
    "prompt": "In the heart of the open sky, Where the winds of change freely sigh, A soul finds its endless flight, In the boundless realms of light.(This prompt will be directly tokenized from plain text)"
}
```

## Citation
If you find this project helps your research, please kindly consider citing our project in your publications.
```
@misc{tang2024tokenizerpruner,
    title = {Qwen Tokenizer Pruner},
    author = {Tang, Kaihua},
    year = {2024},
    note = {\url{https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner}},
}
```