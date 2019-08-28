# pylint: disable=no-self-use, protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from ..modules.language_model_heads.linear import LinearLanguageModelHead  # pylint: disable=unused-import


class TestNextTokenLMPredictor(AllenNlpTestCase):
    def test_predictions_to_labeled_instances(self):
        inputs = {
                "sentence": "Eric Wallace was an intern at",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'next_token_lm' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'next_token_lm')

        instance = predictor._json_to_instance(inputs)
        outputs = predictor._model.forward_on_instance(instance)
        new_instances = predictor.predictions_to_labeled_instances(instance, outputs)
        assert len(new_instances) == 1
        assert 'target_ids' in new_instances[0]
        assert len(new_instances[0]['target_ids'].tokens) == 1  # should have added one word
