# allennlp.models.esim

## ESIM
```python
ESIM(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, similarity_function:allennlp.modules.similarity_functions.similarity_function.SimilarityFunction, projection_feedforward:allennlp.modules.feedforward.FeedForward, inference_encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, output_feedforward:allennlp.modules.feedforward.FeedForward, output_logit:allennlp.modules.feedforward.FeedForward, dropout:float=0.5, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x12f6ca438>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None) -> None
```

This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
<https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
by Chen et al., 2017.

Parameters
----------
vocab : ``Vocabulary``
text_field_embedder : ``TextFieldEmbedder``
    Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
    model.
encoder : ``Seq2SeqEncoder``
    Used to encode the premise and hypothesis.
similarity_function : ``SimilarityFunction``
    This is the similarity function used when computing the similarity matrix between encoded
    words in the premise and words in the hypothesis.
projection_feedforward : ``FeedForward``
    The feedforward network used to project down the encoded and enhanced premise and hypothesis.
inference_encoder : ``Seq2SeqEncoder``
    Used to encode the projected premise and hypothesis for prediction.
output_feedforward : ``FeedForward``
    Used to prepare the concatenated premise and hypothesis for prediction.
output_logit : ``FeedForward``
    This feedforward network computes the output logits.
dropout : ``float``, optional (default=0.5)
    Dropout percentage to use.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.

### forward
```python
ESIM.forward(self, premise:Dict[str, torch.LongTensor], hypothesis:Dict[str, torch.LongTensor], label:torch.IntTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
premise : Dict[str, torch.LongTensor]
    From a ``TextField``
hypothesis : Dict[str, torch.LongTensor]
    From a ``TextField``
label : torch.IntTensor, optional (default = None)
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

