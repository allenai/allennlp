# pylint: disable=no-self-use,invalid-name
from typing import List
from os.path import join

from overrides import overrides

from allennlp.common import Params
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.dataset_readers.squad import _char_span_to_token_span
from allennlp.data.tokenizers import WordTokenizer
from allennlp.testing.test_case import AllenNlpTestCase


class TestSquadReader(AllenNlpTestCase):
    def test_char_span_to_token_span_handles_easy_cases(self):
        tokenizer = WordTokenizer()
        passage = "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy " +\
            "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four " +\
            "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her " +\
            "first performances since giving birth to Blue Ivy."
        tokenized_passage = tokenizer.tokenize(passage)
        # "January 7, 2012"
        token_span = _char_span_to_token_span(passage, tokenized_passage, (3, 18), tokenizer)
        assert token_span == (1, 5)
        # "Lenox Hill Hospital"
        token_span = _char_span_to_token_span(passage, tokenized_passage, (91, 110), tokenizer)
        assert token_span == (22, 25)
        # "Lenox Hill Hospital in New York."
        token_span = _char_span_to_token_span(passage, tokenized_passage, (91, 123), tokenizer)
        assert token_span == (22, 29)

    def test_read_from_file(self):
        reader = SquadReader()
        instances = reader.read('tests/fixtures/squad_example.json').instances
        assert len(instances) == 5
        assert instances[0].fields()["question"].tokens()[:3] == ["To", "whom", "did"]
        assert instances[0].fields()["passage"].tokens()[:3] == ["Architecturally", ",", "the"]
        assert instances[0].fields()["passage"].tokens()[-3:] == ["Mary", ".", "@@STOP@@"]
        assert instances[0].fields()["span_start"].sequence_index() == 102
        assert instances[0].fields()["span_end"].sequence_index() == 105
        assert instances[1].fields()["question"].tokens()[:3] == ["What", "sits", "on"]
        assert instances[1].fields()["passage"].tokens()[:3] == ["Architecturally", ",", "the"]
        assert instances[1].fields()["passage"].tokens()[-3:] == ["Mary", ".", "@@STOP@@"]
        assert instances[1].fields()["span_start"].sequence_index() == 17
        assert instances[1].fields()["span_end"].sequence_index() == 24

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'
