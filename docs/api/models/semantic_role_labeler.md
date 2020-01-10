# allennlp.models.semantic_role_labeler

## SemanticRoleLabeler
```python
SemanticRoleLabeler(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, binary_feature_dim:int, embedding_dropout:float=0.0, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x137274080>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None, label_smoothing:float=None, ignore_span_metric:bool=False, srl_eval_path:str='/Users/markn/allen_ai/allennlp/allennlp/tools/srl-eval.pl') -> None
```

This model performs semantic role labeling using BIO tags using Propbank semantic roles.
Specifically, it is an implementation of `Deep Semantic Role Labeling - What works
and what's next <https://www.aclweb.org/anthology/P17-1044>`_ .

This implementation is effectively a series of stacked interleaved LSTMs with highway
connections, applied to embedded sequences of words concatenated with a binary indicator
containing whether or not a word is the verbal predicate to generate predictions for in
the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
the predictions to contain valid BIO sequences.

Specifically, the model expects and outputs IOB2-formatted tags, where the
B- tag is used in the beginning of every chunk (i.e. all chunks start with the B- tag).

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
encoder : ``Seq2SeqEncoder``
    The encoder (with its own internal stacking) that we will use in between embedding tokens
    and predicting output tags.
binary_feature_dim : int, required.
    The dimensionality of the embedding of the binary verb predicate features.
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
SemanticRoleLabeler.forward(self, tokens:Dict[str, torch.LongTensor], verb_indicator:torch.LongTensor, tags:torch.LongTensor=None, metadata:List[Dict[str, Any]]=None) -> Dict[str, torch.Tensor]
```

Parameters
----------
tokens : Dict[str, torch.LongTensor], required
    The output of ``TextField.as_array()``, which should typically be passed directly to a
    ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
    tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is : ``{"tokens":
    Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
    for the ``TokenIndexers`` when you created the ``TextField`` representing your
    sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
    which knows how to combine different word representations into a single vector per
    token in your input.
verb_indicator: torch.LongTensor, required.
    An integer ``SequenceFeatureField`` representation of the position of the verb
    in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
    all zeros, in the case that the sentence has no verbal predicate.
tags : torch.LongTensor, optional (default = None)
    A torch tensor representing the sequence of integer gold class labels
    of shape ``(batch_size, num_tokens)``
metadata : ``List[Dict[str, Any]]``, optional, (default = None)
    metadata containg the original words in the sentence and the verb to compute the
    frame for, under 'words' and 'verb' keys, respectively.

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
SemanticRoleLabeler.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
``"tags"`` key to the dictionary with the result.

### get_viterbi_pairwise_potentials
```python
SemanticRoleLabeler.get_viterbi_pairwise_potentials(self)
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
SemanticRoleLabeler.get_start_transitions(self)
```

In the BIO sequence, we cannot start the sequence with an I-XXX tag.
This transition sequence is passed to viterbi_decode to specify this constraint.

Returns
-------
start_transitions : torch.Tensor
    The pairwise potentials between a START token and
    the first token of the sequence.

## write_to_conll_eval_file
```python
write_to_conll_eval_file(prediction_file:TextIO, gold_file:TextIO, verb_index:Union[int, NoneType], sentence:List[str], prediction:List[str], gold_labels:List[str])
```

.. deprecated:: 0.8.4
   The ``write_to_conll_eval_file`` function was deprecated in favor of the
   identical ``write_bio_formatted_tags_to_file`` in version 0.8.4.

Prints predicate argument predictions and gold labels for a single verbal
predicate in a sentence to two provided file references.

The CoNLL SRL format is described in
`the shared task data README <https://www.lsi.upc.edu/~srlconll/conll05st-release/README>`_ .

This function expects IOB2-formatted tags, where the B- tag is used in the beginning
of every chunk (i.e. all chunks start with the B- tag).

Parameters
----------
prediction_file : TextIO, required.
    A file reference to print predictions to.
gold_file : TextIO, required.
    A file reference to print gold labels to.
verb_index : Optional[int], required.
    The index of the verbal predicate in the sentence which
    the gold labels are the arguments for, or None if the sentence
    contains no verbal predicate.
sentence : List[str], required.
    The word tokens.
prediction : List[str], required.
    The predicted BIO labels.
gold_labels : List[str], required.
    The gold BIO labels.

