# Qwen-Tokenizer-Pruner
由于Qwen模型的超大vocab_size(151936)，在模型部署和小型化时他的Embedding和LM_Head的权重维度过于巨大，因此本项目提供了一套Qwen和Qwen-VL的Tokenizer词表剪裁方案。

**如果我的开源项目给你带来了启发，给予我一些赞助将对我后续的开源工作有很大帮助。**

[支持我的后续开源工作❤️🙏](https://kaihuatang.github.io/donate.html)

# 代码更新中

```
python main.py --old_model_path XXX --new_model_path XXX --support_data ./data --support_lang 'zh-cn' 'en'
```