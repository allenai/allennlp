from typing import List, Tuple, Callable, TypeVar

from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
from allennlp.data.tokenizers.token import Token

T = TypeVar("T", str, Token)
def enumerate_spans(sentence: List[T],
                    offset: int = 0,
                    max_span_width: int = None,
                    min_span_width: int = 1,
                    filter_function: Callable[[List[T]], bool] = None) -> List[Tuple[int, int]]:
    """
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
    """
    max_span_width = max_span_width or len(sentence)
    filter_function = filter_function or (lambda x: True)
    spans: List[Tuple[int, int]] = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index
            # add 1 to end index because span indices are inclusive.
            if filter_function(sentence[slice(start_index, end_index + 1)]):
                spans.append((start, end))
    return spans


def bio_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.

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
    """
    classes_to_ignore = classes_to_ignore or []
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "U":
            # The U tag is used to indicate a span of length 1,
            # so if there's an active tag we end it, and then
            # we add a "length 0" tag.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            spans.add((conll_tag, (index, index)))
            active_conll_tag = None
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)
