# pylint: disable=no-self-use,invalid-name
import os
import shutil
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestWikiTablesParserPredictor(TestCase):
    def setUp(self):
        super().setUp()
        self.should_remove_data_dir = not os.path.exists('data')

    def tearDown(self):
        super().tearDown()
        if self.should_remove_data_dir and os.path.exists('data'):
            shutil.rmtree('data')

    def test_uses_named_inputs(self):
        inputs = {
                "question": "names",
                "table": "name\tdate\nmatt\t2017\npradeep\t2018"
        }

        archive_dir = 'tests/fixtures/semantic_parsing/wikitables/serialization/'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
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

        archive_dir = 'tests/fixtures/semantic_parsing/wikitables/serialization/'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'wikitables-parser')

        result = predictor.predict_json(inputs)
        answer = result.get("answer")
        assert answer is not None
