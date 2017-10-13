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
        dataset = conll_reader.read('tests/fixtures/data/coref/sample.gold_conll')

        assert len(dataset.instances) == 2

        instances = dataset.instances
        fields = instances[0].fields
        text = [x.text for x in fields["text"].tokens]
        assert text == ['In', 'the', 'summer', 'of', '2005', ',', 'a', 'picture', 'that',
                        'people', 'have', 'long', 'been', 'looking', 'forward', 'to',
                        'started', 'emerging', 'with', 'frequency', 'in', 'various', 'major',
                        'Hong', 'Kong', 'media', '.', 'With', 'their', 'unique', 'charm', ',',
                        'these', 'well', '-', 'known', 'cartoon', 'images', 'once', 'again',
                        'caused', 'Hong', 'Kong', 'to', 'be', 'a', 'focus', 'of', 'worldwide',
                        'attention', '.', 'The', 'world', "'s", 'fifth', 'Disney', 'park',
                        'will', 'soon', 'open', 'to', 'the', 'public', 'here', '.']

        span_starts = fields["span_starts"].field_list
        span_ends = fields["span_ends"].field_list

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove((["Hong", "Kong"], 0))
        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        assert (["their"], 1) in gold_mentions_with_ids
        # This is a span which exceeds our max_span_width, so it should not be considered.
        assert not (["these", "well", "known", "cartoon", "images"], 1) in gold_mentions_with_ids

        fields = instances[1].fields
        text = [x.text for x in fields["text"].tokens]
        assert text == ['The', 'area', 'of', 'Hong', 'Kong', 'is', 'only', 'one', 'thousand', '-', 'plus',
                        'square', 'kilometers', '.', 'The', 'population', 'is', 'dense', '.', 'Natural',
                        'resources', 'are', 'relatively', 'scarce', '.', 'However', ',', 'the', 'clever',
                        'Hong', 'Kong', 'people', 'will', 'utilize', 'all', 'resources', 'they', 'have',
                        'created', 'for', 'developing', 'the', 'Hong', 'Kong', 'tourism', 'industry', '.']

        span_starts = fields["span_starts"].field_list
        span_ends = fields["span_ends"].field_list

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)
        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove((["Hong", "Kong"], 0))
        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        assert (["they"], 1) in gold_mentions_with_ids
        assert (['the', 'clever', 'Hong', 'Kong', 'people'], 1) in gold_mentions_with_ids

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
