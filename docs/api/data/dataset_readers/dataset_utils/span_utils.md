# allennlp.data.dataset_readers.dataset_utils.span_utils

## enumerate_spans
```python
enumerate_spans(sentence:List[~T], offset:int=0, max_span_width:int=None, min_span_width:int=1, filter_function:Callable[[List[~T]], bool]=None) -> List[Tuple[int, int]]
```

Given a sentence, return all token spans within the sentence. Spans are `inclusive`.
Additionally, you can provide a maximum and minimum span width, which will be used
to exclude spans outside of this range.

Finally, you can provide a function mapping ``List[T] -> bool``, which will
be applied to every span to decide whether that span should be included. This
allows filtering by length, regex matches, pos tags or any Spacy ``Token``
attributes, for example.

Parameters
----------
sentence : ``List[T]``, required.
    The sentence to generate spans for. The type is generic, as this function
    can be used with strings, or Spacy ``Tokens`` or other sequences.
offset : ``int``, optional (default = 0)
    A numeric offset to add to all span start and end indices. This is helpful
    if the sentence is part of a larger structure, such as a document, which
    the indices need to respect.
max_span_width : ``int``, optional (default = None)
    The maximum length of spans which should be included. Defaults to len(sentence).
min_span_width : ``int``, optional (default = 1)
    The minimum length of spans which should be included. Defaults to 1.
filter_function : ``Callable[[List[T]], bool]``, optional (default = None)
    A function mapping sequences of the passed type T to a boolean value.
    If ``True``, the span is included in the returned spans from the
    sentence, otherwise it is excluded..

## bio_tags_to_spans
```python
bio_tags_to_spans(tag_sequence:List[str], classes_to_ignore:List[str]=None) -> List[Tuple[str, Tuple[int, int]]]
```

Given a sequence corresponding to BIO tags, extracts spans.
Spans are inclusive and can be of zero length, representing a single word span.
Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
as otherwise it is possible to get a perfect precision score whilst still predicting
ill-formed spans in addition to the correct spans. This function works properly when
the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").

Parameters
----------
tag_sequence : List[str], required.
    The integer class labels for a sequence.
classes_to_ignore : List[str], optional (default = None).
    A list of string class labels `excluding` the bio tag
    which should be ignored when extracting spans.

Returns
-------
spans : List[TypedStringSpan]
    The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
    Note that the label `does not` contain any BIO tag prefixes.

## iob1_tags_to_spans
```python
iob1_tags_to_spans(tag_sequence:List[str], classes_to_ignore:List[str]=None) -> List[Tuple[str, Tuple[int, int]]]
```

Given a sequence corresponding to IOB1 tags, extracts spans.
Spans are inclusive and can be of zero length, representing a single word span.
Ill-formed spans are also included (i.e., those where "B-LABEL" is not preceded
by "I-LABEL" or "B-LABEL").

Parameters
----------
tag_sequence : List[str], required.
    The integer class labels for a sequence.
classes_to_ignore : List[str], optional (default = None).
    A list of string class labels `excluding` the bio tag
    which should be ignored when extracting spans.

Returns
-------
spans : List[TypedStringSpan]
    The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
    Note that the label `does not` contain any BIO tag prefixes.

## bioul_tags_to_spans
```python
bioul_tags_to_spans(tag_sequence:List[str], classes_to_ignore:List[str]=None) -> List[Tuple[str, Tuple[int, int]]]
```

Given a sequence corresponding to BIOUL tags, extracts spans.
Spans are inclusive and can be of zero length, representing a single word span.
Ill-formed spans are not allowed and will raise ``InvalidTagSequence``.
This function works properly when the spans are unlabeled (i.e., your labels are
simply "B", "I", "O", "U", and "L").

Parameters
----------
tag_sequence : ``List[str]``, required.
    The tag sequence encoded in BIOUL, e.g. ["B-PER", "L-PER", "O"].
classes_to_ignore : ``List[str]``, optional (default = None).
    A list of string class labels `excluding` the bio tag
    which should be ignored when extracting spans.

Returns
-------
spans : ``List[TypedStringSpan]``
    The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).

## to_bioul
```python
to_bioul(tag_sequence:List[str], encoding:str='IOB1') -> List[str]
```

Given a tag sequence encoded with IOB1 labels, recode to BIOUL.

In the IOB1 scheme, I is a token inside a span, O is a token outside
a span and B is the beginning of span immediately following another
span of the same type.

In the BIO scheme, I is a token inside a span, O is a token outside
a span and B is the beginning of a span.

Parameters
----------
tag_sequence : ``List[str]``, required.
    The tag sequence encoded in IOB1, e.g. ["I-PER", "I-PER", "O"].
encoding : `str`, optional, (default = ``IOB1``).
    The encoding type to convert from. Must be either "IOB1" or "BIO".

Returns
-------
bioul_sequence : ``List[str]``
    The tag sequence encoded in IOB1, e.g. ["B-PER", "L-PER", "O"].

## bmes_tags_to_spans
```python
bmes_tags_to_spans(tag_sequence:List[str], classes_to_ignore:List[str]=None) -> List[Tuple[str, Tuple[int, int]]]
```

Given a sequence corresponding to BMES tags, extracts spans.
Spans are inclusive and can be of zero length, representing a single word span.
Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
as otherwise it is possible to get a perfect precision score whilst still predicting
ill-formed spans in addition to the correct spans.
This function works properly when the spans are unlabeled (i.e., your labels are
simply "B", "M", "E" and "S").

Parameters
----------
tag_sequence : List[str], required.
    The integer class labels for a sequence.
classes_to_ignore : List[str], optional (default = None).
    A list of string class labels `excluding` the bio tag
    which should be ignored when extracting spans.

Returns
-------
spans : List[TypedStringSpan]
    The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
    Note that the label `does not` contain any BIO tag prefixes.

