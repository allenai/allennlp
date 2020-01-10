# allennlp.modules.seq2vec_encoders.cnn_encoder

## CnnEncoder
```python
CnnEncoder(self, embedding_dim:int, num_filters:int, ngram_filter_sizes:Tuple[int, ...]=(2, 3, 4, 5), conv_layer_activation:allennlp.nn.activations.Activation=None, output_dim:Union[int, NoneType]=None) -> None
```

A ``CnnEncoder`` is a combination of multiple convolution layers and max pooling layers.  As a
:class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
input_dim)``, and the output is of shape ``(batch_size, output_dim)``.

The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
out a vector of size num_filters. The number of times a convolution layer will be used
is ``num_tokens - ngram_size + 1``. The corresponding maxpooling layer aggregates all these
outputs from the convolution layer and outputs the max.

This operation is repeated for every ngram size passed, and consequently the dimensionality of
the output after maxpooling is ``len(ngram_filter_sizes) * num_filters``.  This then gets
(optionally) projected down to a lower dimensional output, specified by ``output_dim``.

We then use a fully connected layer to project in back to the desired output_dim.  For more
details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

Parameters
----------
embedding_dim : ``int``, required
    This is the input dimension to the encoder.  We need this because we can't do shape
    inference in pytorch, and we need to know what size filters to construct in the CNN.
num_filters : ``int``, required
    This is the output dim for each convolutional layer, which is the number of "filters"
    learned by that layer.
ngram_filter_sizes : ``Tuple[int]``, optional (default=``(2, 3, 4, 5)``)
    This specifies both the number of convolutional layers we will create and their sizes.  The
    default of ``(2, 3, 4, 5)`` will have four convolutional layers, corresponding to encoding
    ngrams of size 2 to 5 with some number of filters.
conv_layer_activation : ``Activation``, optional (default=``torch.nn.ReLU``)
    Activation to use after the convolution layers.
output_dim : ``Optional[int]``, optional (default=``None``)
    After doing convolutions and pooling, we'll project the collected features into a vector of
    this size.  If this value is ``None``, we will just return the result of the max pooling,
    giving an output of shape ``len(ngram_filter_sizes) * num_filters``.

### get_input_dim
```python
CnnEncoder.get_input_dim(self) -> int
```

Returns the dimension of the vector input for each element in the sequence input
to a ``Seq2VecEncoder``. This is `not` the shape of the input tensor, but the
last element of that shape.

### get_output_dim
```python
CnnEncoder.get_output_dim(self) -> int
```

Returns the dimension of the final vector output by this ``Seq2VecEncoder``.  This is `not`
the shape of the returned tensor, but the last element of that shape.

