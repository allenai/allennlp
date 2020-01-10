# allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper

## PytorchSeq2VecWrapper
```python
PytorchSeq2VecWrapper(self, module:torch.nn.modules.rnn.RNNBase) -> None
```

Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
the last time step for every layer.  We just want the second one as a single output.  This
wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
set of labels.  The linear layer needs to know its input dimension before it is called, and you
can get that from ``get_output_dim``.

Also, there are lots of ways you could imagine going from an RNN hidden state at every
timestep to a single vector - you could take the last vector at all layers in the stack, do
some kind of pooling, take the last vector of the top layer in a stack, or many other  options.
We just take the final hidden state vector, or in the case of a bidirectional RNN cell, we
concatenate the forward and backward final states together. TODO(mattg): allow for other ways
of wrapping RNNs.

In order to be wrapped with this wrapper, a class must have the following members:

    - ``self.input_size: int``
    - ``self.hidden_size: int``
    - ``def forward(inputs: PackedSequence, hidden_state: torch.tensor) ->
      Tuple[PackedSequence, torch.Tensor]``.
    - ``self.bidirectional: bool`` (optional)

This is what pytorch's RNN's look like - just make sure your class looks like those, and it
should work.

Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
second parameter.

