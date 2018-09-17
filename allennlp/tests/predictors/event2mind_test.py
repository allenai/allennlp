# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestEvent2MindPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "source": "personx gave persony a present",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'event2mind' /
                               'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'event2mind')

        result = predictor.predict_json(inputs)

        token_names = [
                'xintent_top_k_predicted_tokens',
                'xreact_top_k_predicted_tokens',
                'oreact_top_k_predicted_tokens'
        ]

        for token_name in token_names:
            all_predicted_tokens = result.get(token_name)
            for predicted_tokens in all_predicted_tokens:
                assert isinstance(predicted_tokens, list)
                assert all(isinstance(x, str) for x in predicted_tokens)
