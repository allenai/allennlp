# allennlp.models.decomposable_attention

## DecomposableAttention
```python
DecomposableAttention(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, attend_feedforward:allennlp.modules.feedforward.FeedForward, similarity_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction, compare_feedforward:allennlp.modules.feedforward.FeedForward, aggregate_feedforward:allennlp.modules.feedforward.FeedForward, premise_encoder:Union[allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, NoneType]=None, hypothesis_encoder:Union[allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, NoneType]=None, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x12d4d8a20>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Model`` implements the Decomposable Attention model described in `"A Decomposable
Attention Model for Natural Language Inference"
<https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27>`_
by Parikh et al., 2016, with some optional enhancements before the decomposable attention
actually happens.  Parikh's original model allowed for computing an "intra-sentence" attention
before doing the decomposable entailment step.  We generalize this to any
:class:`Seq2SeqEncoder` that can be applied to the premise and/or the hypothesis before
computing entailment.

The basic outline of this model is to get an embedded representation of each word in the
premise and hypothesis, align words between the two, compare the aligned phrases, and make a
final entailment decision based on this aggregated comparison.  Each step in this process uses
a feedforward network to modify the representation.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
    model.
attend_feedforward : ``FeedForward``
    This feedforward network is applied to the encoded sentence representations before the
    similarity matrix is computed between words in the premise and words in the hypothesis.
similarity_function : ``SimilarityFunction``
    This is the similarity function used when computing the similarity matrix between words in
    the premise and words in the hypothesis.
compare_feedforward : ``FeedForward``
    This feedforward network is applied to the aligned premise and hypothesis representations,
    individually.
aggregate_feedforward : ``FeedForward``
    This final feedforward network is applied to the concatenated, summed result of the
    ``compare_feedforward`` network, and its output is used as the entailment class logits.
premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
    After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
    will do nothing.
hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
    After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
    we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
    is also ``None``).
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
DecomposableAttention.forward(self, premise:Dict[str, torch.LongTensor], hypothesis:Dict[str, torch.LongTensor], label:torch.IntTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
premise : Dict[str, torch.LongTensor]
    From a ``TextField``
hypothesis : Dict[str, torch.LongTensor]
    From a ``TextField``
label : torch.IntTensor, optional, (default = None)
    From a ``LabelField``
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    Metadata containing the original tokenization of the premise and
    hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
Returns
-------
An output dictionary consisting of:

label_logits : torch.FloatTensor
    A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
    probabilities of the entailment label.
label_probs : torch.FloatTensor
    A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
    entailment label.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.

