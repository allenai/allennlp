# allennlp.data.token_indexers.wordpiece_indexer

## WordpieceIndexer
```python
WordpieceIndexer(self, vocab:Dict[str, int], wordpiece_tokenizer:Callable[[str], List[str]], namespace:str='wordpiece', use_starting_offsets:bool=False, max_pieces:int=512, do_lowercase:bool=False, never_lowercase:List[str]=None, start_tokens:List[str]=None, end_tokens:List[str]=None, separator_token:str='[SEP]', truncate_long_sequences:bool=True, token_min_padding_length:int=0) -> None
```

A token indexer that does the wordpiece-tokenization (e.g. for BERT embeddings).
If you are using one of the pretrained BERT models, you'll want to use the ``PretrainedBertIndexer``
subclass rather than this base class.

Parameters
----------
vocab : ``Dict[str, int]``
    The mapping {wordpiece -> id}.  Note this is not an AllenNLP ``Vocabulary``.
wordpiece_tokenizer : ``Callable[[str], List[str]]``
    A function that does the actual tokenization.
namespace : str, optional (default: "wordpiece")
    The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
    will be loaded.
use_starting_offsets : bool, optional (default: False)
    By default, the "offsets" created by the token indexer correspond to the
    last wordpiece in each word. If ``use_starting_offsets`` is specified,
    they will instead correspond to the first wordpiece in each word.
max_pieces : int, optional (default: 512)
    The BERT embedder uses positional embeddings and so has a corresponding
    maximum length for its input ids. Any inputs longer than this will
    either be truncated (default), or be split apart and batched using a
    sliding window.
do_lowercase : ``bool``, optional (default=``False``)
    Should we lowercase the provided tokens before getting the indices?
    You would need to do this if you are using an -uncased BERT model
    but your DatasetReader is not lowercasing tokens (which might be the
    case if you're also using other embeddings based on cased tokens).
never_lowercase : ``List[str]``, optional
    Tokens that should never be lowercased. Default is
    ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
start_tokens : ``List[str]``, optional (default=``None``)
    These are prepended to the tokens provided to ``tokens_to_indices``.
end_tokens : ``List[str]``, optional (default=``None``)
    These are appended to the tokens provided to ``tokens_to_indices``.
separator_token : ``str``, optional (default=``[SEP]``)
    This token indicates the segments in the sequence.
truncate_long_sequences : ``bool``, optional (default=``True``)
    By default, long sequences will be truncated to the maximum sequence
    length. Otherwise, they will be split apart and batched using a
    sliding window.
token_min_padding_length : ``int``, optional (default=``0``)
    See :class:`TokenIndexer`.

### count_vocab_items
```python
WordpieceIndexer.count_vocab_items(self, token:allennlp.data.tokenizers.token.Token, counter:Dict[str, Dict[str, int]])
```

The :class:`Vocabulary` needs to assign indices to whatever strings we see in the training
data (possibly doing some frequency filtering and using an OOV, or out of vocabulary,
token).  This method takes a token and a dictionary of counts and increments counts for
whatever vocabulary items are present in the token.  If this is a single token ID
representation, the vocabulary item is likely the token itself.  If this is a token
characters representation, the vocabulary items are all of the characters in the token.

### tokens_to_indices
```python
WordpieceIndexer.tokens_to_indices(self, tokens:List[allennlp.data.tokenizers.token.Token], vocabulary:allennlp.data.vocabulary.Vocabulary, index_name:str) -> Dict[str, List[int]]
```

Takes a list of tokens and converts them to one or more sets of indices.
This could be just an ID for each token from the vocabulary.
Or it could split each token into characters and return one ID per character.
Or (for instance, in the case of byte-pair encoding) there might not be a clean
mapping from individual tokens to indices.

### get_padding_lengths
```python
WordpieceIndexer.get_padding_lengths(self, token:int) -> Dict[str, int]
```

This method returns a padding dictionary for the given token that specifies lengths for
all arrays that need padding.  For example, for single ID tokens the returned dictionary
will be empty, but for a token characters representation, this will return the number
of characters in the token.

### as_padded_tensor
```python
WordpieceIndexer.as_padded_tensor(self, tokens:Dict[str, List[int]], desired_num_tokens:Dict[str, int], padding_lengths:Dict[str, int]) -> Dict[str, torch.Tensor]
```

This method pads a list of tokens to ``desired_num_tokens`` and returns that padded list
of input tokens as a torch Tensor. If the input token list is longer than ``desired_num_tokens``
then it will be truncated.

``padding_lengths`` is used to provide supplemental padding parameters which are needed
in some cases.  For example, it contains the widths to pad characters to when doing
character-level padding.

Note that this method should be abstract, but it is implemented to allow backward compatability.

### get_keys
```python
WordpieceIndexer.get_keys(self, index_name:str) -> List[str]
```

We need to override this because the indexer generates multiple keys.

## PretrainedBertIndexer
```python
PretrainedBertIndexer(self, pretrained_model:str, use_starting_offsets:bool=False, do_lowercase:bool=True, never_lowercase:List[str]=None, max_pieces:int=512, truncate_long_sequences:bool=True) -> None
```

A ``TokenIndexer`` corresponding to a pretrained BERT model.

Parameters
----------
pretrained_model : ``str``
    Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
    or the path to the .txt file with its vocabulary.

    If the name is a key in the list of pretrained models at
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
    the corresponding path will be used; otherwise it will be interpreted as a path or URL.
use_starting_offsets: bool, optional (default: False)
    By default, the "offsets" created by the token indexer correspond to the
    last wordpiece in each word. If ``use_starting_offsets`` is specified,
    they will instead correspond to the first wordpiece in each word.
do_lowercase : ``bool``, optional (default = True)
    Whether to lowercase the tokens before converting to wordpiece ids.
never_lowercase : ``List[str]``, optional
    Tokens that should never be lowercased. Default is
    ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
max_pieces: int, optional (default: 512)
    The BERT embedder uses positional embeddings and so has a corresponding
    maximum length for its input ids. Any inputs longer than this will
    either be truncated (default), or be split apart and batched using a
    sliding window.
truncate_long_sequences : ``bool``, optional (default=``True``)
    By default, long sequences will be truncated to the maximum sequence
    length. Otherwise, they will be split apart and batched using a
    sliding window.

