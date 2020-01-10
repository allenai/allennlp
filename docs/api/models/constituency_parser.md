# allennlp.models.constituency_parser

## SpanInformation
```python
SpanInformation(self, /, *args, **kwargs)
```

A helper namedtuple for handling decoding information.

Parameters
----------
start : ``int``
    The start index of the span.
end : ``int``
    The exclusive end index of the span.
no_label_prob : ``float``
    The probability of this span being assigned the ``NO-LABEL`` label.
label_prob : ``float``
    The probability of the most likely label.

### end
Alias for field number 1
### label_index
Alias for field number 4
### label_prob
Alias for field number 2
### no_label_prob
Alias for field number 3
### start
Alias for field number 0
## SpanConstituencyParser
```python
SpanConstituencyParser(self, vocab:allennlp.data.vocabulary.Vocabulary, text_field_embedder:allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder, span_extractor:allennlp.modules.span_extractors.span_extractor.SpanExtractor, encoder:allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, feedforward:allennlp.modules.feedforward.FeedForward=None, pos_tag_embedding:allennlp.modules.token_embedders.embedding.Embedding=None, initializer:allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x137f933c8>, regularizer:Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType]=None, evalb_directory_path:str='/Users/markn/allen_ai/allennlp/allennlp/tools/EVALB') -> None
```

This ``SpanConstituencyParser`` simply encodes a sequence of text
with a stacked ``Seq2SeqEncoder``, extracts span representations using a
``SpanExtractor``, and then predicts a label for each span in the sequence.
These labels are non-terminal nodes in a constituency parse tree, which we then
greedily reconstruct.

Parameters
----------
vocab : ``Vocabulary``, required
    A Vocabulary, required in order to compute sizes for input/output projections.
text_field_embedder : ``TextFieldEmbedder``, required
    Used to embed the ``tokens`` ``TextField`` we get as input to the model.
span_extractor : ``SpanExtractor``, required.
    The method used to extract the spans from the encoded sequence.
encoder : ``Seq2SeqEncoder``, required.
    The encoder that we will use in between embedding tokens and
    generating span representations.
feedforward : ``FeedForward``, required.
    The FeedForward layer that we will use in between the encoder and the linear
    projection to a distribution over span labels.
pos_tag_embedding : ``Embedding``, optional.
    Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
    Used to initialize the model parameters.
regularizer : ``RegularizerApplicator``, optional (default=``None``)
    If provided, will be used to calculate the regularization penalty during training.
evalb_directory_path : ``str``, optional (default=``DEFAULT_EVALB_DIR``)
    The path to the directory containing the EVALB executable used to score
    bracketed parses. By default, will use the EVALB included with allennlp,
    which is located at allennlp/tools/EVALB . If ``None``, EVALB scoring
    is not used.

### forward
```python
SpanConstituencyParser.forward(self, tokens:Dict[str, torch.LongTensor], spans:torch.LongTensor, metadata:List[Dict[str, Any]], pos_tags:Dict[str, torch.LongTensor]=None, span_labels:torch.LongTensor=None) -> Dict[str, torch.Tensor]
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
spans : ``torch.LongTensor``, required.
    A tensor of shape ``(batch_size, num_spans, 2)`` representing the
    inclusive start and end indices of all possible spans in the sentence.
metadata : List[Dict[str, Any]], required.
    A dictionary of metadata for each batch element which has keys:
        tokens : ``List[str]``, required.
            The original string tokens in the sentence.
        gold_tree : ``nltk.Tree``, optional (default = None)
            Gold NLTK trees for use in evaluation.
        pos_tags : ``List[str]``, optional.
            The POS tags for the sentence. These can be used in the
            model as embedded features, but they are passed here
            in addition for use in constructing the tree.
pos_tags : ``torch.LongTensor``, optional (default = None)
    The output of a ``SequenceLabelField`` containing POS tags.
span_labels : ``torch.LongTensor``, optional (default = None)
    A torch tensor representing the integer gold class labels for all possible
    spans, of shape ``(batch_size, num_spans)``.

Returns
-------
An output dictionary consisting of:
class_probabilities : ``torch.FloatTensor``
    A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
    representing a distribution over the label classes per span.
spans : ``torch.LongTensor``
    The original spans tensor.
tokens : ``List[List[str]]``, required.
    A list of tokens in the sentence for each element in the batch.
pos_tags : ``List[List[str]]``, required.
    A list of POS tags in the sentence for each element in the batch.
num_spans : ``torch.LongTensor``, required.
    A tensor of shape (batch_size), representing the lengths of non-padded spans
    in ``enumerated_spans``.
loss : ``torch.FloatTensor``, optional
    A scalar loss to be optimised.

### decode
```python
SpanConstituencyParser.decode(self, output_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
```

Constructs an NLTK ``Tree`` given the scored spans. We also switch to exclusive
span ends when constructing the tree representation, because it makes indexing
into lists cleaner for ranges of text, rather than individual indices.

Finally, for batch prediction, we will have padded spans and class probabilities.
In order to make this less confusing, we remove all the padded spans and
distributions from ``spans`` and ``class_probabilities`` respectively.

### construct_trees
```python
SpanConstituencyParser.construct_trees(self, predictions:torch.FloatTensor, all_spans:torch.LongTensor, num_spans:torch.LongTensor, sentences:List[List[str]], pos_tags:List[List[str]]=None) -> List[nltk.tree.Tree]
```

Construct ``nltk.Tree``'s for each batch element by greedily nesting spans.
The trees use exclusive end indices, which contrasts with how spans are
represented in the rest of the model.

Parameters
----------
predictions : ``torch.FloatTensor``, required.
    A tensor of shape ``(batch_size, num_spans, span_label_vocab_size)``
    representing a distribution over the label classes per span.
all_spans : ``torch.LongTensor``, required.
    A tensor of shape (batch_size, num_spans, 2), representing the span
    indices we scored.
num_spans : ``torch.LongTensor``, required.
    A tensor of shape (batch_size), representing the lengths of non-padded spans
    in ``enumerated_spans``.
sentences : ``List[List[str]]``, required.
    A list of tokens in the sentence for each element in the batch.
pos_tags : ``List[List[str]]``, optional (default = None).
    A list of POS tags for each word in the sentence for each element
    in the batch.

Returns
-------
A ``List[Tree]`` containing the decoded trees for each element in the batch.

### resolve_overlap_conflicts_greedily
```python
SpanConstituencyParser.resolve_overlap_conflicts_greedily(spans:List[allennlp.models.constituency_parser.SpanInformation]) -> List[allennlp.models.constituency_parser.SpanInformation]
```

Given a set of spans, removes spans which overlap by evaluating the difference
in probability between one being labeled and the other explicitly having no label
and vice-versa. The worst case time complexity of this method is ``O(k * n^4)`` where ``n``
is the length of the sentence that the spans were enumerated from (and therefore
``k * m^2`` complexity with respect to the number of spans ``m``) and ``k`` is the
number of conflicts. However, in practice, there are very few conflicts. Hopefully.

This function modifies ``spans`` to remove overlapping spans.

Parameters
----------
spans : ``List[SpanInformation]``, required.
    A list of spans, where each span is a ``namedtuple`` containing the
    following attributes:

    start : ``int``
        The start index of the span.
    end : ``int``
        The exclusive end index of the span.
    no_label_prob : ``float``
        The probability of this span being assigned the ``NO-LABEL`` label.
    label_prob : ``float``
        The probability of the most likely label.

Returns
-------
A modified list of ``spans``, with the conflicts resolved by considering local
differences between pairs of spans and removing one of the two spans.

### construct_tree_from_spans
```python
SpanConstituencyParser.construct_tree_from_spans(spans_to_labels:Dict[Tuple[int, int], str], sentence:List[str], pos_tags:List[str]=None) -> nltk.tree.Tree
```

Parameters
----------
spans_to_labels : ``Dict[Tuple[int, int], str]``, required.
    A mapping from spans to constituency labels.
sentence : ``List[str]``, required.
    A list of tokens forming the sentence to be parsed.
pos_tags : ``List[str]``, optional (default = None)
    A list of the pos tags for the words in the sentence, if they
    were either predicted or taken as input to the model.

Returns
-------
An ``nltk.Tree`` constructed from the labelled spans.

### get_metrics
```python
SpanConstituencyParser.get_metrics(self, reset:bool=False) -> Dict[str, float]
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

