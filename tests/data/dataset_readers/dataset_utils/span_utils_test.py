from typing import List
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import span_utils
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers.token import Token


class SpanUtilsTest(AllenNlpTestCase):
    def test_bio_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {
            ("ARG1", (1, 2)),
            ("ARG2", (4, 5)),
            ("ARG1", (6, 6)),
            ("ARG2", (7, 7)),
        }

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "U-ARG1", "U-ARG2"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.bio_tags_to_spans(tag_sequence)

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = [
            "O",
            "B-ARG1",
            "I-ARG1",
            "O",
            "I-ARG1",
            "B-ARG2",
            "I-ARG2",
            "B-ARG1",
            "I-ARG2",
            "I-ARG2",
        ]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {
            ("ARG1", (1, 2)),
            ("ARG2", (5, 6)),
            ("ARG1", (7, 7)),
            ("ARG1", (4, 4)),
            ("ARG2", (8, 9)),
        }

    def test_bio_tags_to_spans_extracts_correct_spans_without_labels(self):
        tag_sequence = ["O", "B", "I", "O", "B", "I", "B", "B"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (1, 2)), ("", (4, 5)), ("", (6, 6)), ("", (7, 7))}

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = ["O", "B", "I", "O", "B", "I", "U", "U"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.bio_tags_to_spans(tag_sequence)

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = ["O", "B", "I", "O", "I", "B", "I", "B", "I", "I"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (1, 2)), ("", (4, 4)), ("", (5, 6)), ("", (7, 9))}

    def test_bio_tags_to_spans_ignores_specified_tags(self):
        tag_sequence = [
            "B-V",
            "I-V",
            "O",
            "B-ARG1",
            "I-ARG1",
            "O",
            "B-ARG2",
            "I-ARG2",
            "B-ARG1",
            "B-ARG2",
        ]
        spans = span_utils.bio_tags_to_spans(tag_sequence, ["ARG1", "V"])
        assert set(spans) == {("ARG2", (6, 7)), ("ARG2", (9, 9))}

    def test_iob1_tags_to_spans_extracts_correct_spans_without_labels(self):
        tag_sequence = ["I", "B", "I", "O", "B", "I", "B", "B"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 2)), ("", (4, 5)), ("", (6, 6)), ("", (7, 7))}

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = ["O", "B", "I", "O", "B", "I", "U", "U"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.iob1_tags_to_spans(tag_sequence)

        # Check that invalid IOB1 sequences are also handled as spans.
        tag_sequence = ["O", "B", "I", "O", "I", "B", "I", "B", "I", "I"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (1, 2)), ("", (4, 4)), ("", (5, 6)), ("", (7, 9))}

    def test_iob1_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = ["I-ARG2", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == {
            ("ARG2", (0, 0)),
            ("ARG1", (1, 2)),
            ("ARG2", (4, 5)),
            ("ARG1", (6, 6)),
            ("ARG2", (7, 7)),
        }

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "U-ARG1", "U-ARG2"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.iob1_tags_to_spans(tag_sequence)

        # Check that invalid IOB1 sequences are also handled as spans.
        tag_sequence = [
            "O",
            "B-ARG1",
            "I-ARG1",
            "O",
            "I-ARG1",
            "B-ARG2",
            "I-ARG2",
            "B-ARG1",
            "I-ARG2",
            "I-ARG2",
        ]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == {
            ("ARG1", (1, 2)),
            ("ARG1", (4, 4)),
            ("ARG2", (5, 6)),
            ("ARG1", (7, 7)),
            ("ARG2", (8, 9)),
        }

    def test_enumerate_spans_enumerates_all_spans(self):
        tokenizer = SpacyTokenizer(pos_tags=True)
        sentence = tokenizer.tokenize("This is a sentence.")

        spans = span_utils.enumerate_spans(sentence)
        assert spans == [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4),
        ]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2)
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2, offset=20)
        assert spans == [(20, 21), (20, 22), (21, 22), (21, 23), (22, 23), (22, 24), (23, 24)]

        def no_prefixed_punctuation(tokens: List[Token]):
            # Only include spans which don't start or end with punctuation.
            return tokens[0].pos_ != "PUNCT" and tokens[-1].pos_ != "PUNCT"

        spans = span_utils.enumerate_spans(
            sentence, max_span_width=3, min_span_width=2, filter_function=no_prefixed_punctuation
        )

        # No longer includes (2, 4) or (3, 4) as these include punctuation
        # as their last element.
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    def test_bioul_tags_to_spans(self):
        tag_sequence = ["B-PER", "I-PER", "L-PER", "U-PER", "U-LOC", "O"]
        spans = span_utils.bioul_tags_to_spans(tag_sequence)
        assert spans == [("PER", (0, 2)), ("PER", (3, 3)), ("LOC", (4, 4))]

        tag_sequence = ["B-PER", "I-PER", "O"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.bioul_tags_to_spans(tag_sequence)

    def test_bioul_tags_to_spans_without_labels(self):
        tag_sequence = ["B", "I", "L", "U", "U", "O"]
        spans = span_utils.bioul_tags_to_spans(tag_sequence)
        assert spans == [("", (0, 2)), ("", (3, 3)), ("", (4, 4))]

        tag_sequence = ["B", "I", "O"]
        with pytest.raises(span_utils.InvalidTagSequence):
            spans = span_utils.bioul_tags_to_spans(tag_sequence)

    def test_iob1_to_bioul(self):
        tag_sequence = ["I-ORG", "O", "I-MISC", "O"]
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding="IOB1")
        assert bioul_sequence == ["U-ORG", "O", "U-MISC", "O"]

        tag_sequence = ["O", "I-PER", "B-PER", "I-PER", "I-PER", "B-PER"]
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding="IOB1")
        assert bioul_sequence == ["O", "U-PER", "B-PER", "I-PER", "L-PER", "U-PER"]

    def test_bio_to_bioul(self):
        tag_sequence = ["B-ORG", "O", "B-MISC", "O", "B-MISC", "I-MISC", "I-MISC"]
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding="BIO")
        assert bioul_sequence == ["U-ORG", "O", "U-MISC", "O", "B-MISC", "I-MISC", "L-MISC"]

        # Encoding in IOB format should throw error with incorrect encoding.
        with pytest.raises(span_utils.InvalidTagSequence):
            tag_sequence = ["O", "I-PER", "B-PER", "I-PER", "I-PER", "B-PER"]
            bioul_sequence = span_utils.to_bioul(tag_sequence, encoding="BIO")

    def test_bmes_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = ["B-ARG1", "M-ARG1", "E-ARG1", "B-ARG2", "E-ARG2", "S-ARG3"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 2)), ("ARG2", (3, 4)), ("ARG3", (5, 5))}

        tag_sequence = ["S-ARG1", "B-ARG2", "E-ARG2", "S-ARG3"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 0)), ("ARG2", (1, 2)), ("ARG3", (3, 3))}

        # Invalid labels.
        tag_sequence = ["B-ARG1", "M-ARG2"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 0)), ("ARG2", (1, 1))}

        tag_sequence = ["B-ARG1", "E-ARG2"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 0)), ("ARG2", (1, 1))}

        tag_sequence = ["B-ARG1", "M-ARG1", "M-ARG2"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 1)), ("ARG2", (2, 2))}

        tag_sequence = ["B-ARG1", "M-ARG1", "E-ARG2"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 1)), ("ARG2", (2, 2))}

        # Invalid transitions.
        tag_sequence = ["B-ARG1", "B-ARG1"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 0)), ("ARG1", (1, 1))}

        tag_sequence = ["B-ARG1", "S-ARG1"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (0, 0)), ("ARG1", (1, 1))}

    def test_bmes_tags_to_spans_extracts_correct_spans_without_labels(self):
        # Good transitions.
        tag_sequence = ["B", "M", "E", "B", "E", "S"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 2)), ("", (3, 4)), ("", (5, 5))}

        tag_sequence = ["S", "B", "E", "S"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 2)), ("", (3, 3))}

        # Invalid transitions.
        tag_sequence = ["B", "B", "E"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 2))}

        tag_sequence = ["B", "S"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 1))}

        tag_sequence = ["M", "B", "E"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 2))}

        tag_sequence = ["B", "M", "S"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 1)), ("", (2, 2))}

        tag_sequence = ["B", "E", "M", "E"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 1)), ("", (2, 3))}

        tag_sequence = ["B", "E", "E"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 1)), ("", (2, 2))}

        tag_sequence = ["S", "M"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 1))}

        tag_sequence = ["S", "E"]
        spans = span_utils.bmes_tags_to_spans(tag_sequence)
        assert set(spans) == {("", (0, 0)), ("", (1, 1))}
