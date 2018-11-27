# pylint: disable=no-self-use,invalid-name
import os
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.predictors.wikitables_parser import (SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH)

@pytest.mark.java
class TestWikiTablesParserPredictor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

    def tearDown(self):
        super().tearDown()
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)

    def test_uses_named_inputs(self):
        inputs = {
                "question": "names",
                "table": "name\tdate\nmatt\t2017\npradeep\t2018"
        }

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'wikitables-parser')

        result = predictor.predict_json(inputs)

        action_sequence = result.get("best_action_sequence")
        if action_sequence:
            # We don't currently disallow endless loops in the decoder, and an untrained seq2seq
            # model will easily get itself into a loop.  An endless loop isn't a finished logical
            # form, so decoding doesn't return any finished states, which means no actions.  So,
            # sadly, we don't have a great test here.  This is just testing that the predictor
            # runs, basically.
            assert len(action_sequence) > 1
            assert all([isinstance(action, str) for action in action_sequence])

            logical_form = result.get("logical_form")
            assert logical_form is not None

    def test_answer_present(self):
        inputs = {
                "question": "Who is 18 years old?",
                "table": "Name\tAge\nShallan\t16\nKaladin\t18"
        }

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'wikitables-parser')

        result = predictor.predict_json(inputs)
        answer = result.get("answer")
        assert answer is not None

    def test_answer_present_with_batch_predict(self):
        inputs = [{
                "question": "Who is 18 years old?",
                "table": "Name\tAge\nShallan\t16\nKaladin\t18"
        }]

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'wikitables' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'wikitables-parser')

        result = predictor.predict_batch_json(inputs)
        answer = result[0].get("answer")
        assert answer is not None
