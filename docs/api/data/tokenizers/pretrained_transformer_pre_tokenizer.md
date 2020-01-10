# allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer

## OpenAIPreTokenizer
```python
OpenAIPreTokenizer(self, language:str='en_core_web_sm') -> None
```

For OpenAI transformer
This is used to split a sentence into words.
Then the ``OpenaiTransformerBytePairIndexer`` converts each word into wordpieces.

### batch_tokenize
```python
OpenAIPreTokenizer.batch_tokenize(self, texts:List[str]) -> List[List[allennlp.data.tokenizers.token.Token]]
```

Batches together tokenization of several texts, in case that is faster for particular
tokenizers.

By default we just do this without batching.  Override this in your tokenizer if you have a
good way of doing batched computation.

### tokenize
```python
OpenAIPreTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

## BertPreTokenizer
```python
BertPreTokenizer(self, do_lower_case:bool=True, never_split:Union[List[str], NoneType]=None) -> None
```

The ``BasicTokenizer`` from the BERT implementation.
This is used to split a sentence into words.
Then the ``BertTokenIndexer`` converts each word into wordpieces.

### tokenize
```python
BertPreTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

Actually implements splitting words into tokens.

Returns
-------
tokens : ``List[Token]``

