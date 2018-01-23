# pylint: disable=no-self-use,invalid-name,protected-access
from typing import List

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import span_utils
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.data.tokenizers.token import Token

class SpanUtilsTest(AllenNlpTestCase):

    def test_bio_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (4, 5)), ("ARG1", (6, 6)), ("ARG2", (7, 7))}

        # Check that it works when we use U- tags for single tokens.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "U-ARG1", "U-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (4, 5)), ("ARG1", (6, 6)), ("ARG2", (7, 7))}

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "I-ARG1", "B-ARG2", "I-ARG2", "B-ARG1", "I-ARG2", "I-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (5, 6)), ("ARG1", (7, 7)),
                              ("ARG1", (4, 4)), ("ARG2", (8, 9))}

    def test_bio_tags_to_spans_ignores_specified_tags(self):
        tag_sequence = ["B-V", "I-V", "O", "B-ARG1", "I-ARG1",
                        "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence, ["ARG1", "V"])
        assert set(spans) == {("ARG2", (6, 7)), ("ARG2", (9, 9))}

    def test_enumerate_spans_enumerates_all_spans(self):
        tokenizer = SpacyWordSplitter(pos_tags=True)
        sentence = tokenizer.split_words("This is a sentence.")

        spans = span_utils.enumerate_spans(sentence)
        assert spans == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2),
                         (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2)
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2, offset=20)
        assert spans == [(20, 21), (20, 22), (21, 22), (21, 23), (22, 23), (22, 24), (23, 24)]

        def no_prefixed_punctuation(tokens: List[Token]):
            # Only include spans which don't start or end with punctuation.
            return tokens[0].pos_ != "PUNCT" and tokens[-1].pos_ != "PUNCT"

        spans = span_utils.enumerate_spans(sentence,
                                           max_span_width=3,
                                           min_span_width=2,
                                           filter_function=no_prefixed_punctuation)

        # No longer includes (2, 4) or (3, 4) as these include punctuation
        # as their last element.
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
