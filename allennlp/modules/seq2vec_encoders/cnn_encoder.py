from typing import Optional, Tuple

from overrides import overrides
import torch
from torch.nn import Conv1d, Linear

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype


@Seq2VecEncoder.register("cnn")
class CnnEncoder(Seq2VecEncoder):
    """
    A `CnnEncoder` is a combination of multiple convolution layers and max pooling layers.  As a
    [`Seq2VecEncoder`](./seq2vec_encoder.md), the input to this module is of shape `(batch_size, num_tokens,
    input_dim)`, and the output is of shape `(batch_size, output_dim)`.

    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filters`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.

    Registered as a `Seq2VecEncoder` with name "cnn".

    # Parameters

    embedding_dim : `int`, required
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters : `int`, required
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes : `Tuple[int]`, optional (default=`(2, 3, 4, 5)`)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation : `Activation`, optional (default=`torch.nn.ReLU`)
        Activation to use after the convolution layers.
    output_dim : `Optional[int]`, optional (default=`None`)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is `None`, we will just return the result of the max pooling,
        giving an output of shape `len(ngram_filter_sizes) * num_filters`.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        conv_layer_activation: Activation = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name("relu")()

        self._convolution_layers = [
            Conv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size,
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if output_dim:
            self.projection_layer = Linear(maxpool_output_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()

        # Our input is expected to have shape `(batch_size, num_tokens, embedding_dim)`.  The
        # convolution layers expect input of shape `(batch_size, in_channels, sequence_length)`,
        # where the conv layer `in_channels` is our `embedding_dim`.  We thus need to transpose the
        # tensor first.
        tokens = torch.transpose(tokens, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length)`,
        # where `pool_length = num_tokens - ngram_size + 1`.  We then do an activation function,
        # masking, then do max pooling over each filter for the whole input sequence.
        # Because our max pooling is simple, we just use `torch.max`.  The resultant tensor has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        # To ensure the cnn_encoder respects masking we add a large negative value to
        # the activations of all filters that convolved over a masked token. We do this by
        # first enumerating all filters for a given convolution size (torch.arange())
        # then by comparing it to an index of the last filter that does not involve a masked
        # token (.ge()) and finally adjusting dimensions to allow for addition and multiplying
        # by a large negative value (.unsqueeze())
        filter_outputs = []
        batch_size = tokens.shape[0]
        # shape: (batch_size, 1)
        last_unmasked_tokens = mask.sum(dim=1).unsqueeze(dim=-1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = tokens.shape[2] - convolution_layer.kernel_size[0] + 1

            # Forward pass of the convolutions.
            # shape: (batch_size, num_filters, pool_length)
            activations = self._activation(convolution_layer(tokens))

            # Create activation mask.
            # shape: (batch_size, pool_length)
            indices = (
                torch.arange(pool_length, device=activations.device)
                .unsqueeze(0)
                .expand(batch_size, pool_length)
            )
            # shape: (batch_size, pool_length)
            activations_mask = indices.ge(
                last_unmasked_tokens - convolution_layer.kernel_size[0] + 1
            )
            # shape: (batch_size, num_filters, pool_length)
            activations_mask = activations_mask.unsqueeze(1).expand_as(activations)

            # Replace masked out values with smallest possible value of the dtype so
            # that max pooling will ignore these activations.
            # shape: (batch_size, pool_length)
            activations = activations + (activations_mask * min_value_of_dtype(activations.dtype))

            # Pick out the max filters
            filter_outputs.append(activations.max(dim=2)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = (
            torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        )

        # Replace the maxpool activations that picked up the masks with 0s
        maxpool_output[maxpool_output == min_value_of_dtype(maxpool_output.dtype)] = 0.0

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result
