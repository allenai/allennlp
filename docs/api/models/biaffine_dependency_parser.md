# allennlp.models.biaffine_dependency_parser

## BiaffineDependencyParser
```python
BiaffineDependencyParser(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, tag_representation_dim:int, arc_representation_dim:int, tag_feedforward:allennlp.modules.feedforward.FeedForward=None, arc_feedforward:allennlp.modules.feedforward.FeedForward=None, pos_tag_embedding:allennlp.modules.token_embedders.embedding.Embedding=None, use_mst_decoding_for_validation:bool=True, dropout:float=0.0, input_dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x1398630f0>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This dependency parser follows the model of
` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
<https://arxiv.org/abs/1611.01734>`_ .

Word representations are generated using a bidirectional LSTM,
followed by separate biaffine classifiers for pairs of words,
predicting whether a directed arc exists between the two words
and the dependency label the arc should have. Decoding can either
be done greedily, or the optimal Minimum Spanning Tree can be
decoded using Edmond's algorithm by viewing the dependency tree as
a MST on a fully connected graph, where nodes are words and edges
are scored dependency arcs.

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
encoder : ``Seq2SeqEncoder``
    The encoder (with its own internal stacking) that we will use to generate representations
    of tokens.
tag_representation_dim : ``int``, required.
    The dimension of the MLPs used for dependency tag prediction.
arc_representation_dim : ``int``, required.
    The dimension of the MLPs used for head arc prediction.
tag_feedforward : ``FeedForward``, optional, (default = None).
    The feedforward network used to produce tag representations.
    By default, a 1 layer feedforward network with an elu activation is used.
arc_feedforward : ``FeedForward``, optional, (default = None).
    The feedforward network used to produce arc representations.
    By default, a 1 layer feedforward network with an elu activation is used.
pos_tag_embedding : ``Embedding``, optional.
    Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
use_mst_decoding_for_validation : ``bool``, optional (default = True).
    Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
    If false, decoding is greedy.
dropout : ``float``, optional, (default = 0.0)
    The variational dropout applied to the output of the encoder and MLP layers.
input_dropout : ``float``, optional, (default = 0.0)
    The dropout applied to the embedded text input.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
BiaffineDependencyParser.forward(self, words:Dict[str, torch.LongTensor], pos_tags:torch.LongTensor, metadata:List[Dict[str, Any]], head_tags:torch.LongTensor=None, head_indices:torch.LongTensor=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
words : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``, which should typically be passed directly to a
    ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
    tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is : ``{"tokens":
    Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
    for the ``TokenIndexers`` when you created the ``TextField`` representing your
    sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
    which knows how to combine different word representations into a single vector per
    token in your input.
pos_tags : ``torch.LongTensor``, required
    The output of a ``SequenceLabelField`` containing POS tags.
    POS tags are required regardless of whether they are used in the model,
    because they are used to filter the evaluation metric to only consider
    heads of words which are not punctuation.
metadata : List[Dict[str, Any]], optional (default=None)
    A dictionary of metadata for each batch element which has keys:
        words : ``List[str]``, required.
            The tokens in the original sentence.
        pos : ``List[str]``, required.
            The dependencies POS tags for each word.
head_tags : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer gold class labels for the arcs
    in the dependency parse. Has shape ``(batch_size, sequence_length)``.
head_indices : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer indices denoting the parent of every
    word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

Returns
-------
An output dictionary consisting of:
loss : ``torch.FloatTensor``, optional
    A scalar loss to be optimised.
arc_loss : ``torch.FloatTensor``
    The loss contribution from the unlabeled arcs.
loss : ``torch.FloatTensor``, optional
    The loss contribution from predicting the dependency
    tags for the gold arcs.
heads : ``torch.FloatTensor``
    The predicted head indices for each word. A tensor
    of shape (batch_size, sequence_length).
head_types : ``torch.FloatTensor``
    The predicted head types for each arc. A tensor
    of shape (batch_size, sequence_length).
mask : ``torch.LongTensor``
    A mask denoting the padded elements in the batch.

### decode
```python
BiaffineDependencyParser.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Takes the result of :func:`forward` and runs inference / decoding / whatever
post-processing you need to do your model.  The intent is that ``model.forward()`` should
produce potentials or probabilities, and then ``model.decode()`` can take those results and
run some kind of beam search or constrained inference or whatever is necessary.  This does
not handle all possible decoding use cases, but it at least handles simple kinds of
decoding.

This method `modifies` the input dictionary, and also `returns` the same dictionary.

By default in the base class we do nothing.  If your model has some special decoding step,
override this method.

### get_metrics
```python
BiaffineDependencyParser.get_metrics(self, reset:bool=False) -> Dict[str, float]
```

Returns a dictionary of metrics. This method will be called by
:class:`allennlp.training.Trainer` in order to compute and use model metrics for early
stopping and model serialization.  We return an empty dictionary here rather than raising
as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
passed, as frequently a metric accumulator will have some state which should be reset
between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
should be populated during the call to ``forward``, with the
:class:`~allennlp.training.Metric` handling the accumulation of the metric until this
method is called.

