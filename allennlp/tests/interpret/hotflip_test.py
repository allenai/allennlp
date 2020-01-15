from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import Hotflip
from allennlp.interpret.attackers.hotflip import DEFAULT_IGNORE_TOKENS

from ..modules.language_model_heads.linear import LinearLanguageModelHead


class TestHotflip(AllenNlpTestCase):
    def test_hotflip(self):
        inputs = {
            "premise": "I always write unit tests for my code.",
            "hypothesis": "One time I didn't write any unit tests for my code.",
        }

        archive = load_archive(
            self.FIXTURES_ROOT / "decomposable_attention" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "textual-entailment")

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, "hypothesis", "grad_input_1")
        assert attack is not None
        assert "final" in attack
        assert "original" in attack
        assert "outputs" in attack
        assert len(attack["final"][0]) == len(
            attack["original"]
        )  # hotflip replaces words without removing

    def test_targeted_attack_from_json(self):
        inputs = {"sentence": "The doctor ran to the emergency room to see [MASK] patient."}

        archive = load_archive(
            self.FIXTURES_ROOT / "masked_language_model" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "masked_language_model")

        hotflipper = Hotflip(predictor, vocab_namespace="tokens")
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, target={"words": ["hi"]})
        assert attack is not None
        assert "final" in attack
        assert "original" in attack
        assert "outputs" in attack
        assert len(attack["final"][0]) == len(
            attack["original"]
        )  # hotflip replaces words without removing
        assert attack["final"][0] != attack["original"]
