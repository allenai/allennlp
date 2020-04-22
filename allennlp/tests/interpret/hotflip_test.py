from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.interpret.attackers import Hotflip
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import EmptyEmbedder
from allennlp.predictors import Predictor


class TestHotflip(AllenNlpTestCase):
    def test_hotflip(self):
        inputs = {"sentence": "I always write unit tests for my code."}

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive)

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, "tokens", "grad_input_1")
        assert attack is not None
        assert "final" in attack
        assert "original" in attack
        assert "outputs" in attack
        assert len(attack["final"][0]) == len(
            attack["original"]
        )  # hotflip replaces words without removing

    def test_with_token_characters_indexer(self):

        inputs = {"sentence": "I always write unit tests for my code."}

        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive)
        predictor._dataset_reader._token_indexers["chars"] = TokenCharactersIndexer(
            min_padding_length=1
        )
        predictor._model._text_field_embedder._token_embedders["chars"] = EmptyEmbedder()

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, "tokens", "grad_input_1")
        assert attack is not None
        assert "final" in attack
        assert "original" in attack
        assert "outputs" in attack
        assert len(attack["final"][0]) == len(
            attack["original"]
        )  # hotflip replaces words without removing
