# allennlp.models.srl_bert

## SrlBert
```python
SrlBert(self, vocab:allennlp.data.vocabulary.Vocabulary, bert_model:Union[str, pytorch_pretrained_bert.modeling.BertModel], embedding_dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x139b8dc88>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None, label_smoothing:float=None, ignore_span_metric:bool=False, srl_eval_path:str='/Users/markn/allen_ai/allennlp/allennlp/tools/srl-eval.pl') -> None
```


Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
model : ``Union[str, BertModel]``, required.
    A string describing the BERT model to load or an already constructed BertModel.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.
label_smoothing : ``float``, optional (default = 0.0)
    Whether or not to use label smoothing on the labels when computing cross entropy loss.
ignore_span_metric : ``bool``, optional (default = False)
    Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
srl_eval_path : ``str``, optional (default=``DEFAULT_SRL_EVAL_PATH``)
    The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
    which is located at allennlp/tools/srl-eval.pl . If ``None``, srl-eval.pl is not used.

### forward
```python
SrlBert.forward(self, tokens:Dict[str, torch.Tensor], verb_indicator:torch.Tensor, metadata:List[Any], tags:torch.LongTensor=None)
```

Parameters
----------
tokens : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``, which should typically be passed directly to a
    ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
    indexes wordpieces from the BERT vocabulary.
verb_indicator: torch.LongTensor, required.
    An integer ``SequenceFeatureField`` representation of the position of the verb
    in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
    all zeros, in the case that the sentence has no verbal predicate.
tags : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer gold class labels
    of shape ``(batch_size, num_tokens)``
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    metadata containg the original words in the sentence, the verb to compute the
    frame for, and start offsets for converting wordpieces back to a sequence of words,
    under 'words', 'verb' and 'offsets' keys, respectively.

Returns
-------
An output dictionary consisting of:
logits : torch.FloatTensor
    A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
    unnormalised log probabilities of the tag classes.
class_probabilities : torch.FloatTensor
    A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
    a distribution of the tag classes per word.
loss : torch.FloatTensor, optional
    A scalar loss to be optimised.

### decode
```python
SrlBert.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
``"tags"`` key to the dictionary with the result.

NOTE: First, we decode a BIO sequence on top of the wordpieces. This is important; viterbi
decoding produces low quality output if you decode on top of word representations directly,
because the model gets confused by the 'missing' positions (which is sensible as it is trained
to perform tagging on wordpieces, not words).

Secondly, it's important that the indices we use to recover words from the wordpieces are the
start_offsets (i.e offsets which correspond to using the first wordpiece of words which are
tokenized into multiple wordpieces) as otherwise, we might get an ill-formed BIO sequence
when we select out the word tags from the wordpiece tags. This happens in the case that a word
is split into multiple word pieces, and then we take the last tag of the word, which might
correspond to, e.g, I-V, which would not be allowed as it is not preceeded by a B tag.

### get_viterbi_pairwise_potentials
```python
SrlBert.get_viterbi_pairwise_potentials(self)
```

Generate a matrix of pairwise transition potentials for the BIO labels.
The only constraint implemented here is that I-XXX labels must be preceded
by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
constraint, pairs of labels which do not satisfy this constraint have a
pairwise potential of -inf.

Returns
-------
transition_matrix : torch.Tensor
    A (num_labels, num_labels) matrix of pairwise potentials.

### get_start_transitions
```python
SrlBert.get_start_transitions(self)
```

In the BIO sequence, we cannot start the sequence with an I-XXX tag.
This transition sequence is passed to viterbi_decode to specify this constraint.

Returns
-------
start_transitions : torch.Tensor
    The pairwise potentials between a START token and
    the first token of the sequence.

