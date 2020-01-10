# allennlp.data.tokenizers.pretrained_transformer_tokenizer

## PretrainedTransformerTokenizer
```python
PretrainedTransformerTokenizer(self, model_name:str, add_special_tokens:bool=True, max_length:int=None, stride:int=0, truncation_strategy:str='longest_first', calculate_character_offsets:bool=False) -> None
```

A ``PretrainedTransformerTokenizer`` uses a model from HuggingFace's
``transformers`` library to tokenize some input text.  This often means wordpieces
(where ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is',
'awesome']``), but it could also use byte-pair encoding, or some other tokenization, depending
on the pretrained model that you're using.

We take a model name as an input parameter, which we will pass to
``AutoTokenizer.from_pretrained``.

We also add special tokens relative to the pretrained model and truncate the sequences.

This tokenizer also indexes tokens and adds the indexes to the ``Token`` fields so that
they can be picked up by ``PretrainedTransformerIndexer``.

Parameters
----------
model_name : ``str``
    The name of the pretrained wordpiece tokenizer to use.
add_special_tokens : ``bool``, optional, (default=True)
    If set to ``True``, the sequences will be encoded with the special tokens relative
    to their model.
max_length : ``int``, optional (default=None)
    If set to a number, will limit the total sequence returned so that it has a maximum length.
    If there are overflowing tokens, those will be added to the returned dictionary
stride : ``int``, optional (default=0)
    If set to a number along with max_length, the overflowing tokens returned will contain some tokens
    from the main sequence returned. The value of this argument defines the number of additional tokens.
truncation_strategy : ``str``, optional (default='longest_first')
    String selected in the following options:
    - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
    starting from the longest one at each token (when there is a pair of input sequences)
    - 'only_first': Only truncate the first sequence
    - 'only_second': Only truncate the second sequence
    - 'do_not_truncate': Do not truncate (raise an error if the input sequence is longer than max_length)
calculate_character_offsets : ``bool``, optional (default=False)
    Attempts to reconstruct character offsets for the instances of Token that this tokenizer produces.

Argument descriptions are from
https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691

### tokenize_sentence_pair
```python
PretrainedTransformerTokenizer.tokenize_sentence_pair(self, sentence_1:str, sentence_2:str) -> List[allennlp.data.tokenizers.token.Token]
```

This methods properly handles a pair of sentences.

### tokenize
```python
PretrainedTransformerTokenizer.tokenize(self, text:str) -> List[allennlp.data.tokenizers.token.Token]
```

This method only handles a single sentence (or sequence) of text.
Refer to the ``tokenize_sentence_pair`` method if you have a sentence pair.

