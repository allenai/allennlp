# allennlp.models.biaffine_dependency_parser_multilang

## BiaffineDependencyParserMultiLang
```python
BiaffineDependencyParserMultiLang(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, tag_representation_dim:int, arc_representation_dim:int, tag_feedforward:allennlp.modules.feedforward.FeedForward=None, arc_feedforward:allennlp.modules.feedforward.FeedForward=None, pos_tag_embedding:allennlp.modules.token_embedders.embedding.Embedding=None, use_mst_decoding_for_validation:bool=True, langs_for_early_stop:List[str]=None, dropout:float=0.0, input_dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x134a630b8>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This dependency parser implements the multi-lingual extension
of the Dozat and Manning (2016) model as described in
`Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot
Dependency Parsing (Schuster et al., 2019) <https://www.aclweb.org/anthology/papers/N/N19/N19-1162>`_ .
Also, please refer to the `alignment computation code
<https://github.com/TalSchuster/CrossLingualELMo>`_.

All parameters are shared across all languages except for
the text_field_embedder. For aligned ELMo embeddings, use the
elmo_token_embedder_multilang with the pre-computed alignments
to the mutual embedding space.
Also, the universal_dependencies_multilang dataset reader
supports loading of multiple sources and storing the language
identifier in the metadata.


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
langs_for_early_stop : ``List[str]``, optional, (default = [])
    Which languages to include in the averaged metrics
    (that could be used for early stopping).
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
BiaffineDependencyParserMultiLang.forward(self, words:Dict[str, torch.LongTensor], pos_tags:torch.LongTensor, metadata:List[Dict[str, Any]], head_tags:torch.LongTensor=None, head_indices:torch.LongTensor=None) -> Dict[str, torch.Tensor]
```

Embedding each language by the corresponding parameters for
``TextFieldEmbedder``. Batches should contain only samples from a
single language.
Metadata should have a ``lang`` key.

### get_metrics
```python
BiaffineDependencyParserMultiLang.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

