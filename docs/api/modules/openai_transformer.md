# allennlp.modules.openai_transformer

An implementation of the OpenAI Transformer Language Model.

Mostly just a slightly modified version of
https://github.com/huggingface/pytorch-openai-transformer-lm
so thanks to them!

Some of these modules duplicate code elsewhere in AllenNLP,
but the serialized weights depend on the exact parameter setup
here, so it's easiest to just reimplement them.

## TransformerConfig
```python
TransformerConfig(self, /, *args, **kwargs)
```

The transformer has to pass a bunch of params to its submodules,
this bundles them together to make things easier.

### activation_function
Alias for field number 5
### attention_dropout_probability
Alias for field number 3
### embedding_dim
Alias for field number 0
### embedding_dropout_probability
Alias for field number 2
### num_heads
Alias for field number 1
### residual_dropout_probability
Alias for field number 4
## LayerNorm
```python
LayerNorm(self, n_state, e=1e-05)
```
Construct a layernorm module in the OpenAI style (epsilon inside the square root).
## OpenaiTransformer
```python
OpenaiTransformer(self, vocab_size:int=40478, n_ctx:int=512, embedding_dim:int=768, num_heads:int=12, num_layers:int=12, embedding_dropout_probability:float=0.1, attention_dropout_probability:float=0.1, residual_dropout_probability:float=0.1, activation_function:str='gelu', model_path:str=None, requires_grad:bool=False, n_special:int=-1) -> None
```

Openai transformer, as per https://blog.openai.com/language-unsupervised/.
Default parameters are the ones for their pretrained model.

Parameters
----------
vocab_size : ``int`` (optional, default: 40478)
    The size of the vocabulary (number of byte pair embeddings)
    excluding the n_special embeddings (if any), and the positional embeddings.
n_ctx : ``int`` (optional, default: 512)
    The number of positional encodings to use for evaluation.
embedding_dim : ``int`` (optional, default: 768)
    The dimension of the output embeddings.
num_heads : ``int`` (optional, default: 12)
    How many "heads" the attention has.
num_layers : ``int`` (optional, default: 12)
    How many layers of "blocks" the transformer has.
embedding_dropout_probability : ``float`` (optional, default: 0.1)
    Dropout for the embedding.
attention_dropout_probability : ``float`` (optional, default: 0.1)
    Dropout for attention.
residual_dropout_probability : ``float`` (optional, default: 0.1)
    Dropout for residual
activation_function : ``str`` (optional, default : ``'gelu'``)
    Activation function for the multi-layer perceptron.
model_path : ``str`` (optional, default : ``None``)
    A tar.gz file containing serialized model weights. If supplied,
    the weights will be loaded from that file.
requires_grad : ``bool`` (optional, default : ``False``)
    If true, the transformer will be fine-tuneable.
n_special : ``int`` (optional, default : ``-1``)
    The number of special tokens added to the byte pair vocabulary
    (via ``OpenaiTransformerBytePairIndexer``).

