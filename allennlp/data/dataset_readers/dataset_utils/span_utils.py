from typing import List, Tuple, Callable, TypeVar

from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
from allennlp.data.tokenizers.token import Token


class InvalidTagSequence(Exception):
    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return ' '.join(self.tag_sequence)


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
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
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


def bioul_tags_to_spans(tag_sequence: List[str],
                        classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIOUL tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are not allowed and will raise ``InvalidTagSequence``.

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
    """
    spans = []
    classes_to_ignore = classes_to_ignore or []
    index = 0
    while index < len(tag_sequence):
        label = tag_sequence[index]
        if label[0] == 'U':
            spans.append((label.partition('-')[2], (index, index)))
        elif label[0] == 'B':
            start = index
            while label[0] != 'L':
                index += 1
                if index >= len(tag_sequence):
                    raise InvalidTagSequence(tag_sequence)
                label = tag_sequence[index]
                if not (label[0] == 'I' or label[0] == 'L'):
                    raise InvalidTagSequence(tag_sequence)
            spans.append((label.partition('-')[2], (start, index)))
        else:
            if label != 'O':
                raise InvalidTagSequence(tag_sequence)
        index += 1
    return [span for span in spans if span[0] not in classes_to_ignore]


def iob1_to_bioul(tag_sequence: List[str]) -> List[str]:
    """
    Given a tag sequence encoded with IOB1 labels, recode to BIOUL.

    In the IOB1 scheme, I is a token inside a span, O is a token outside
    a span and B is the beginning of span immediately following another
    span of the same type.

    Parameters
    ----------
    tag_sequence : ``List[str]``, required.
        The tag sequence encoded in IOB1, e.g. ["I-PER", "I-PER", "O"].

    Returns
    -------
    bioul_sequence: ``List[str]``
        The tag sequence encoded in IOB1, e.g. ["B-PER", "L-PER", "O"].
    """
    # pylint: disable=len-as-condition

    def replace_label(full_label, new_label):
        # example: full_label = 'I-PER', new_label = 'U', returns 'U-PER'
        parts = list(full_label.partition('-'))
        parts[0] = new_label
        return ''.join(parts)

    def pop_replace_append(in_stack, out_stack, new_label):
        # pop the last element from in_stack, replace the label, append
        # to out_stack
        tag = in_stack.pop()
        new_tag = replace_label(tag, new_label)
        out_stack.append(new_tag)

    def process_stack(stack, out_stack):
        # process a stack of labels, add them to out_stack
        if len(stack) == 1:
            # just a U token
            pop_replace_append(stack, out_stack, 'U')
        else:
            # need to code as BIL
            recoded_stack = []
            pop_replace_append(stack, recoded_stack, 'L')
            while len(stack) >= 2:
                pop_replace_append(stack, recoded_stack, 'I')
            pop_replace_append(stack, recoded_stack, 'B')
            recoded_stack.reverse()
            out_stack.extend(recoded_stack)


    # Process the tag_sequence one tag at a time, adding spans to a stack,
    # then recode them.
    bioul_sequence = []
    stack: List[str] = []

    for label in tag_sequence:
        # need to make a dict like
        # token = {'token': 'Matt', "labels": {'conll2003': "B-PER"}
        #                   'gold': 'I-PER'}
        # where 'gold' is the raw value from the CoNLL data set

        if label == 'O' and len(stack) == 0:
            bioul_sequence.append(label)
        elif label == 'O' and len(stack) > 0:
            # need to process the entries on the stack plus this one
            process_stack(stack, bioul_sequence)
            bioul_sequence.append(label)
        elif label[0] == 'I':
            # check if the previous type is the same as this one
            # if it is then append to stack
            # otherwise this start a new entity if the type
            # is different
            if len(stack) == 0:
                stack.append(label)
            else:
                # check if the previous type is the same as this one
                this_type = label.partition('-')[2]
                prev_type = stack[-1].partition('-')[2]
                if this_type == prev_type:
                    stack.append(label)
                else:
                    # a new entity
                    process_stack(stack, bioul_sequence)
                    stack.append(label)
        elif label[0] == 'B':
            if len(stack) > 0:
                process_stack(stack, bioul_sequence)
            stack.append(label)
        else:
            raise InvalidTagSequence(tag_sequence)

    # process the stack
    if len(stack) > 0:
        process_stack(stack, bioul_sequence)

    return bioul_sequence
