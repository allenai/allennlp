# allennlp.modules.language_model_heads.gpt2

## Gpt2LanguageModelHead
```python
Gpt2LanguageModelHead(self, model_name:str) -> None
```

Loads just the LM head from ``transformers.GPT2LMHeadModel``.  It was easiest to load
the entire model before only pulling out the head, so this is a bit slower than it could be,
but for practical use in a model, the few seconds of extra loading time is probably not a big
deal.

