# allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer

The BidirectionalTransformerEncoder from Calypso.
This is basically the transformer from https://nlp.seas.harvard.edu/2018/04/03/attention.html
so credit to them.

This code should be considered "private" in that we have several
transformer implementations and may end up deleting this one.
If you use it, consider yourself warned.

## attention
```python
attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor=None, dropout:Callable=None) -> Tuple[torch.Tensor, torch.Tensor]
```
Compute 'Scaled Dot Product Attention'
## subsequent_mask
```python
subsequent_mask(size:int, device:str='cpu') -> torch.Tensor
```
Mask out subsequent positions.
## PositionalEncoding
```python
PositionalEncoding(self, input_dim:int, max_len:int=5000) -> None
```
Implement the Positional Encoding function.
## PositionwiseFeedForward
```python
PositionwiseFeedForward(self, input_dim:int, ff_dim:int, dropout:float=0.1) -> None
```
Implements FFN equation.
## TransformerEncoder
```python
TransformerEncoder(self, layer:torch.nn.modules.module.Module, num_layers:int, return_all_layers:bool=False) -> None
```
Core encoder is a stack of N layers
### forward
```python
TransformerEncoder.forward(self, x, mask)
```
Pass the input (and mask) through each layer in turn.
## SublayerConnection
```python
SublayerConnection(self, size:int, dropout:float) -> None
```

A residual connection followed by a layer norm.
Note for code simplicity the norm is first as opposed to last.

### forward
```python
SublayerConnection.forward(self, x:torch.Tensor, sublayer:Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor
```
Apply residual connection to any sublayer with the same size.
## EncoderLayer
```python
EncoderLayer(self, size:int, self_attn:torch.nn.modules.module.Module, feed_forward:torch.nn.modules.module.Module, dropout:float) -> None
```
Encoder is made up of self-attn and feed forward (defined below)
### forward
```python
EncoderLayer.forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor
```
Follow Figure 1 (left) for connections.
## make_model
```python
make_model(num_layers:int=6, input_size:int=512, hidden_size:int=2048, heads:int=8, dropout:float=0.1, return_all_layers:bool=False) -> allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer.TransformerEncoder
```
Helper: Construct a model from hyperparameters.
## BidirectionalLanguageModelTransformer
```python
BidirectionalLanguageModelTransformer(self, input_dim:int, hidden_dim:int, num_layers:int, dropout:float=0.1, input_dropout:float=None, return_all_layers:bool=False) -> None
```

### get_attention_masks
```python
BidirectionalLanguageModelTransformer.get_attention_masks(self, mask:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```

Returns 2 masks of shape (batch_size, timesteps, timesteps) representing
1) non-padded elements, and
2) elements of the sequence which are permitted to be involved in attention at a given timestep.

