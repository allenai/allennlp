# allennlp.interpret.attackers.hotflip

## Hotflip
```python
Hotflip(self, predictor:allennlp.predictors.predictor.Predictor, vocab_namespace:str='tokens', max_tokens:int=5000) -> None
```

Runs the HotFlip style attack at the word-level https://arxiv.org/abs/1712.06751.  We use the
first-order taylor approximation described in https://arxiv.org/abs/1903.06620, in the function
``_first_order_taylor()``.

We try to re-use the embedding matrix from the model when deciding what other words to flip a
token to.  For a large class of models, this is straightforward.  When there is a
character-level encoder, however (e.g., with ELMo, any char-CNN, etc.), or a combination of
encoders (e.g., ELMo + glove), we need to construct a fake embedding matrix that we can use in
``_first_order_taylor()``.  We do this by getting a list of words from the model's vocabulary
and embedding them using the encoder.  This can be expensive, both in terms of time and memory
usage, so we take a ``max_tokens`` parameter to limit the size of this fake embedding matrix.
This also requires a model to `have` a token vocabulary in the first place, which can be
problematic for models that only have character vocabularies.

Parameters
----------
predictor : ``Predictor``
    The model (inside a Predictor) that we're attacking.  We use this to get gradients and
    predictions.
vocab_namespace : ``str``, optional (default='tokens')
    We use this to know three things: (1) which tokens we should ignore when producing flips
    (we don't consider non-alphanumeric tokens); (2) what the string value is of the token that
    we produced, so we can show something human-readable to the user; and (3) if we need to
    construct a fake embedding matrix, we use the tokens in the vocabulary as flip candidates.
max_tokens : ``int``, optional (default=5000)
    This is only used when we need to construct a fake embedding matrix.  That matrix can take
    a lot of memory when the vocab size is large.  This parameter puts a cap on the number of
    tokens to use, so the fake embedding matrix doesn't take as much memory.

### initialize
```python
Hotflip.initialize(self)
```

Call this function before running attack_from_json(). We put the call to
``_construct_embedding_matrix()`` in this function to prevent a large amount of compute
being done when __init__() is called.

### attack_from_json
```python
Hotflip.attack_from_json(self, inputs:Dict[str, Any], input_field_to_attack:str='tokens', grad_input_field:str='grad_input_1', ignore_tokens:List[str]=None, target:Dict[str, Any]=None) -> Dict[str, Any]
```

Replaces one token at a time from the input until the model's prediction changes.
``input_field_to_attack`` is for example ``tokens``, it says what the input field is
called.  ``grad_input_field`` is for example ``grad_input_1``, which is a key into a grads
dictionary.

The method computes the gradient w.r.t. the tokens, finds the token with the maximum
gradient (by L2 norm), and replaces it with another token based on the first-order Taylor
approximation of the loss.  This process is iteratively repeated until the prediction
changes.  Once a token is replaced, it is not flipped again.

Parameters
----------
inputs : ``JsonDict``
    The model inputs, the same as what is passed to a ``Predictor``.
input_field_to_attack : ``str``, optional (default='tokens')
    The field that has the tokens that we're going to be flipping.  This must be a
    ``TextField``.
grad_input_field : ``str``, optional (default='grad_input_1')
    If there is more than one field that gets embedded in your model (e.g., a question and
    a passage, or a premise and a hypothesis), this tells us the key to use to get the
    correct gradients.  This selects from the output of :func:`Predictor.get_gradients`.
ignore_tokens : ``List[str]``, optional (default=DEFAULT_IGNORE_TOKENS)
    These tokens will not be flipped.  The default list includes some simple punctuation,
    OOV and padding tokens, and common control tokens for BERT, etc.
target : ``JsonDict``, optional (default=None)
    If given, this will be a `targeted` hotflip attack, where instead of just trying to
    change a model's prediction from what it current is predicting, we try to change it to
    a `specific` target value.  This is a ``JsonDict`` because it needs to specify the
    field name and target value.  For example, for a masked LM, this would be something
    like ``{"words": ["she"]}``, because ``"words"`` is the field name, there is one mask
    token (hence the list of length one), and we want to change the prediction from
    whatever it was to ``"she"``.

