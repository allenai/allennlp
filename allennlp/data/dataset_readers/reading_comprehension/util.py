"""
Utilities for reading comprehension dataset readers.
"""

from collections import Counter, defaultdict
import logging
import string
from typing import Any, Dict, List, Tuple

from allennlp.data.fields import Field, TextField, IndexField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# These are tokens and characters that are stripped by the standard SQuAD and TriviaQA evaluation
# scripts.
IGNORED_TOKENS = {'a', 'an', 'the'}
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])

def normalize_text(text: str) -> str:
    """
    Performs a normalization that is very similar to that done by the normalization functions in
    SQuAD and TriviaQA.

    This involves splitting and rejoining the text, and could be a somewhat expensive operation.
    """
    return ' '.join([token
                     for token in text.lower().strip(STRIPPED_CHARACTERS).split()
                     if token not in IGNORED_TOKENS])


def char_span_to_token_span(token_offsets: List[Tuple[int, int]],
                            character_span: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
    """
    Converts a character span from a passage into the corresponding token span in the tokenized
    version of the passage.  If you pass in a character span that does not correspond to complete
    tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
    We return an error flag in this case, and have some debug logging so you can figure out the
    cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
    problems; there's a fair amount of both).

    The basic outline of this method is to find the token span that has the same offsets as the
    input character span.  If the tokenizer tokenized the passage correctly and has matching
    offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
    but mostly just find the closest thing we can.

    The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
    So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
    two-word span beginning at token index 3, and so on.

    Returns
    -------
    token_span : ``Tuple[int, int]``
        `Inclusive` span start and end token indices that match as closely as possible to the input
        character spans.
    error : ``bool``
        Whether the token spans match the input character spans exactly.  If this is ``False``, it
        means there was an error in either the tokenization or the annotated character span.
    """
    # We have token offsets into the passage from the tokenizer; we _should_ be able to just find
    # the tokens that have the same offsets as our span.
    error = False
    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
        start_index += 1
    # start_index should now be pointing at the span start index.
    if token_offsets[start_index][0] > character_span[0]:
        # In this case, a tokenization or labeling issue made us go too far - the character span
        # we're looking for actually starts in the previous token.  We'll back up one.
        logger.debug("Bad labelling or tokenization - start offset doesn't match")
        start_index -= 1
    if token_offsets[start_index][0] != character_span[0]:
        error = True
    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
        end_index += 1
    if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
        # Looks like there was a token that should have been split, like "1854-1855", where the
        # answer is "1854".  We can't do much in this case, except keep the answer as the whole
        # token.
        logger.debug("Bad tokenization - end offset doesn't match")
    elif token_offsets[end_index][1] > character_span[1]:
        # This is a case where the given answer span is more than one token, and the last token is
        # cut off for some reason, like "split with Luckett and Rober", when the original passage
        # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
        # where it is, and assume the intent was to mark the whole token.
        logger.debug("Bad labelling or tokenization - end offset doesn't match")
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def find_valid_answer_spans(passage_tokens: List[Token],
                            answer_texts: List[str]) -> List[Tuple[int, int]]:
    """
    Finds a list of token spans in ``passage_tokens`` that match the given ``answer_texts``.  This
    tries to find all spans that would evaluate to correct given the SQuAD and TriviaQA official
    evaluation scripts, which do some normalization of the input text.

    Note that this could return duplicate spans!  The caller is expected to be able to handle
    possible duplicates (as already happens in the SQuAD dev set, for instance).
    """
    normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
    # Because there could be many `answer_texts`, we'll do the most expensive pre-processing
    # step once.  This gives us a map from tokens to the position in the passage they appear.
    word_positions: Dict[str, List[int]] = defaultdict(list)
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i)
    spans = []
    for answer_text in answer_texts:
        # For each answer, we'll first find all valid start positions in the passage.  Then
        # we'll grow each span to the same length as the number of answer tokens, and see if we
        # have a match.  We're a little tricky as we grow the span, skipping words that are
        # already pruned from the normalized answer text, and stopping early if we don't match.
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        num_answer_tokens = len(answer_tokens)
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index] == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index:
                spans.append((span_start, span_end))
    return spans


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        answer_texts: List[str] = None,
                                        additional_metadata: Dict[str, Any] = None) -> Instance:
    """
    Converts a question, a passage, and an optional answer (or answers) to an ``Instance`` for use
    in a reading comprehension model.

    Creates an ``Instance`` with at least these fields: ``question`` and ``passage``, both
    ``TextFields``; and ``metadata``, a ``MetadataField``.  Additionally, if both ``answer_texts``
    and ``char_span_starts`` are given, the ``Instance`` has ``span_start`` and ``span_end``
    fields, which are both ``IndexFields``.

    Parameters
    ----------
    question_tokens : ``List[Token]``
        An already-tokenized question.
    passage_tokens : ``List[Token]``
        An already-tokenized passage that contains the answer to the given question.
    token_indexers : ``Dict[str, TokenIndexer]``
        Determines how the question and passage ``TextFields`` will be converted into tensors that
        get input to a model.  See :class:`TokenIndexer`.
    passage_text : ``str``
        The original passage text.  We need this so that we can recover the actual span from the
        original passage that the model predicts as the answer to the question.  This is used in
        official evaluation scripts.
    token_spans : ``List[Tuple[int, int]]``, optional
        Indices into ``passage_tokens`` to use as the answer to the question for training.  This is
        a list because there might be several possible correct answer spans in the passage.
        Currently, we just select the most frequent span in this list (i.e., SQuAD has multiple
        annotations on the dev set; this will select the span that the most annotators gave as
        correct).
    answer_texts : ``List[str]``, optional
        All valid answer strings for the given question.  In SQuAD, e.g., the training set has
        exactly one answer per question, but the dev and test sets have several.  TriviaQA has many
        possible answers, which are the aliases for the known correct entity.  This is put into the
        metadata for use with official evaluation scripts, but not used anywhere else.
    additional_metadata : ``Dict[str, Any]``, optional
        The constructed ``metadata`` field will by default contain ``original_passage``,
        ``token_offsets``, ``question_tokens``, ``passage_tokens``, and ``answer_texts`` keys.  If
        you want any other metadata to be associated with each instance, you can pass that in here.
        This dictionary will get added to the ``metadata`` dictionary we already construct.
    """
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    metadata = {
            'original_passage': passage_text,
            'token_offsets': passage_offsets,
            'question_tokens': [token.text for token in question_tokens],
            'passage_tokens': [token.text for token in passage_tokens],
            }
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    if token_spans:
        # There may be multiple answer annotations, so we pick the one that occurs the most.  This
        # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
        # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
        # all of the annotations.  This is why we have a separate official SQuAD metric calculation
        # (the "em" and "f1" metrics use the official script).
        candidate_answers: Counter = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        fields['span_start'] = IndexField(span_start, passage_field)
        fields['span_end'] = IndexField(span_end, passage_field)

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)
