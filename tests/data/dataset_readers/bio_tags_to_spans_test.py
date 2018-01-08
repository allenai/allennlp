# pylint: disable=no-self-use,invalid-name,protected-access

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils.ontonotes import bio_tags_to_spans

class BioTagsToSpansTest(AllenNlpTestCase):

    def test_bio_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (4, 5)), ("ARG1", (6, 6)), ("ARG2", (7, 7))}

        # Check that it works when we use U- tags for single tokens.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "U-ARG1", "U-ARG2"]
        spans = bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (4, 5)), ("ARG1", (6, 6)), ("ARG2", (7, 7))}

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "I-ARG1", "B-ARG2", "I-ARG2", "B-ARG1", "I-ARG2", "I-ARG2"]
        spans = bio_tags_to_spans(tag_sequence)
        assert set(spans) == {("ARG1", (1, 2)), ("ARG2", (5, 6)), ("ARG1", (7, 7)),
                              ("ARG1", (4, 4)), ("ARG2", (8, 9))}

    def test_bio_tags_to_spans_ignores_specified_tags(self):
        tag_sequence = ["B-V", "I-V", "O", "B-ARG1", "I-ARG1",
                        "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        spans = bio_tags_to_spans(tag_sequence, ["ARG1", "V"])
        assert set(spans) == {("ARG2", (6, 7)), ("ARG2", (9, 9))}
