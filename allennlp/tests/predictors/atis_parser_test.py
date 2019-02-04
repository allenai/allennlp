# pylint: disable=no-self-use,invalid-name
from flaky import flaky

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class TestAtisParserPredictor(AllenNlpTestCase):
    @flaky
    def test_atis_parser_uses_named_inputs(self):
        inputs = {
                "utterance": "show me the flights to seattle",
        }

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'atis' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'atis-parser')

        result = predictor.predict_json(inputs)
        action_sequence = result.get("best_action_sequence")
        if action_sequence:
            # An untrained model will likely get into a loop, and not produce at finished states.
            # When the model gets into a loop it will not produce any valid SQL, so we don't get
            # any actions. This basically just tests if the model runs.
            assert len(action_sequence) > 1
            assert all([isinstance(action, str) for action in action_sequence])
            predicted_sql_query = result.get("predicted_sql_query")
            assert predicted_sql_query is not None

    @flaky
    def test_atis_parser_predicted_sql_present(self):
        inputs = {
                "utterance": "show me flights to seattle"
        }

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'atis' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'atis-parser')

        result = predictor.predict_json(inputs)
        predicted_sql_query = result.get("predicted_sql_query")
        assert predicted_sql_query is not None

    @flaky
    def test_atis_parser_batch_predicted_sql_present(self):
        inputs = [{
                "utterance": "show me flights to seattle",
        }]

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'atis' / 'serialization' / 'model.tar.gz'
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'atis-parser')

        result = predictor.predict_batch_json(inputs)
        predicted_sql_query = result[0].get("predicted_sql_query")
        assert predicted_sql_query is not None
