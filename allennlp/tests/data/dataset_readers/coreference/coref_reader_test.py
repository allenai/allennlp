from typing import List, Tuple

import pytest

from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestCorefReader:
    span_width = 5

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = ConllCorefReader(max_span_width=self.span_width, lazy=lazy)
        instances = ensure_list(
            conll_reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / "coref" / "coref.gold_conll"))
        )

        assert len(instances) == 4

        fields = instances[0].fields
        text = [x.text for x in fields["text"].tokens]

        assert text == [
            "In",
            "the",
            "summer",
            "of",
            "2005",
            ",",
            "a",
            "picture",
            "that",
            "people",
            "have",
            "long",
            "been",
            "looking",
            "forward",
            "to",
            "started",
            "emerging",
            "with",
            "frequency",
            "in",
            "various",
            "major",
            "Hong",
            "Kong",
            "media",
            ".",
            "With",
            "their",
            "unique",
            "charm",
            ",",
            "these",
            "well",
            "-",
            "known",
            "cartoon",
            "images",
            "once",
            "again",
            "caused",
            "Hong",
            "Kong",
            "to",
            "be",
            "a",
            "focus",
            "of",
            "worldwide",
            "attention",
            ".",
            "The",
            "world",
            "'s",
            "fifth",
            "Disney",
            "park",
            "will",
            "soon",
            "open",
            "to",
            "the",
            "public",
            "here",
            ".",
        ]

        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(
            span_starts, span_ends, text
        )

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [
            (candidate_mentions[i], x) for i, x in gold_indices_with_ids
        ]

        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove((["Hong", "Kong"], 0))
        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        assert (["their"], 1) in gold_mentions_with_ids
        # This is a span which exceeds our max_span_width, so it should not be considered.
        assert (
            ["these", "well", "-", "known", "cartoon", "images"],
            1,
        ) not in gold_mentions_with_ids

        fields = instances[2].fields
        text = [x.text for x in fields["text"].tokens]
        assert text == [
            "The",
            "area",
            "of",
            "Hong",
            "Kong",
            "is",
            "only",
            "one",
            "thousand",
            "-",
            "plus",
            "square",
            "kilometers",
            ".",
            "The",
            "population",
            "is",
            "dense",
            ".",
            "Natural",
            "resources",
            "are",
            "relatively",
            "scarce",
            ".",
            "However",
            ",",
            "the",
            "clever",
            "Hong",
            "Kong",
            "people",
            "will",
            "utilize",
            "all",
            "resources",
            "they",
            "have",
            "created",
            "for",
            "developing",
            "the",
            "Hong",
            "Kong",
            "tourism",
            "industry",
            ".",
        ]

        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(
            span_starts, span_ends, text
        )

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [
            (candidate_mentions[i], x) for i, x in gold_indices_with_ids
        ]

        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove((["Hong", "Kong"], 0))
        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        assert (["they"], 1) in gold_mentions_with_ids
        assert (["the", "clever", "Hong", "Kong", "people"], 1) in gold_mentions_with_ids

    def test_wordpiece_modeling(self):
        tokenizer = PretrainedTransformerTokenizer("bert-base-cased")
        conll_reader = ConllCorefReader(
            max_span_width=self.span_width, wordpiece_modeling_tokenizer=tokenizer
        )
        instances = ensure_list(
            conll_reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / "coref" / "coref.gold_conll"))
        )

        assert len(instances) == 4

        fields = instances[3].fields
        text = [x.text for x in fields["text"].tokens]

        assert text == [
            "[CLS]",
            "Hong",
            "Kong",
            "Wet",
            "##land",
            "Park",
            ",",
            "which",
            "is",
            "currently",
            "under",
            "construction",
            ",",
            "is",
            "also",
            "one",
            "of",
            "the",
            "designated",
            "new",
            "projects",
            "of",
            "the",
            "Hong",
            "Kong",
            "SA",
            "##R",
            "government",
            "for",
            "advancing",
            "the",
            "Hong",
            "Kong",
            "tourism",
            "industry",
            ".",
            "[SEP]",
        ]

        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(
            span_starts, span_ends, text
        )

        # Asserts special tokens aren't included in the spans
        assert all(span_start > 0 for span_start in span_starts)
        assert all(span_end < len(text) - 1 for span_end in span_ends)

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [
            (candidate_mentions[i], x) for i, x in gold_indices_with_ids
        ]

        assert (["Hong", "Kong"], 0) in gold_mentions_with_ids
        # Within span_width before wordpiece splitting but exceeds afterwards
        assert (["the", "Hong", "Kong", "SA", "##R", "government"], 0) not in gold_mentions_with_ids

        fields = instances[1].fields
        text = [x.text for x in fields["text"].tokens]
        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])
        candidate_mentions = self.check_candidate_mentions_are_well_defined(
            span_starts, span_ends, text
        )

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [
            (candidate_mentions[i], x) for i, x in gold_indices_with_ids
        ]

        # Prior to wordpiece tokenization, 's was one token; wordpiece tokenization splits it into 2
        assert (["the", "city", "'", "s"], 0) in gold_mentions_with_ids

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in zip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start : end + 1]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all(self.span_width >= len(x) > 0 for x in candidate_mentions)
        return candidate_mentions
