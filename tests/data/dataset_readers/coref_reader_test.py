# pylint: disable=no-self-use,invalid-name

from typing import List, Tuple
from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.common.testing import AllenNlpTestCase


class TestCorefReader(AllenNlpTestCase):

    def setUp(self):
        super(TestCorefReader, self).setUp()
        self.span_width = 5

    def test_read_from_file(self):

        conll_reader = ConllCorefReader(max_span_width=self.span_width)
        dataset = conll_reader.read('tests/fixtures/conll_2012/')

        assert len(dataset.instances) == 1

        instances = dataset.instances
        fields = instances[0].fields
        text = [x.text for x in fields["text"].tokens]

        assert text == ['Mali', 'government', 'officials', 'say', 'the', 'woman', "'s",
                        'confession', 'was', 'forced', '.', 'The', 'prosecution', 'rested',
                        'its', 'case', 'last', 'month', 'after', 'four', 'months', 'of',
                        'hearings', '.', 'Denise', 'Dillon', 'Headline', 'News', '.', 'and',
                        'that', 'wildness', 'is', 'still', 'in', 'him', ',', 'as', 'it', 'is',
                        'with', 'all', 'children', '.']

        span_starts = fields["span_starts"].field_list
        span_ends = fields["span_ends"].field_list

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert gold_mentions_with_ids == [(['the', 'woman', "'s"], 0),
                                          (['the', 'woman', "'s", 'confession'], 1),
                                          (['The', 'prosecution'], 2),
                                          (['its'], 2),
                                          (['Denise', 'Dillon'], 2),
                                          (['him'], 3)]

        # Now check that we don't collect spans greater than the max width.
        conll_reader = ConllCorefReader(max_span_width=2)
        dataset = conll_reader.read('tests/fixtures/conll_2012/')

        instances = dataset.instances
        fields = instances[0].fields
        text = [x.text for x in fields["text"].tokens]
        span_starts = fields["span_starts"].field_list
        span_ends = fields["span_ends"].field_list

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert gold_mentions_with_ids == [(['The', 'prosecution'], 2),
                                          (['its'], 2),
                                          (['Denise', 'Dillon'], 2),
                                          (['him'], 3)]

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in zip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start.sequence_index: end.sequence_index + 1]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all([self.span_width >= len(x) > 0 for x in candidate_mentions])  # pylint: disable=len-as-condition
        return candidate_mentions
