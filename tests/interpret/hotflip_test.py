from pytest import raises
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.interpret.attackers import Hotflip
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import EmptyEmbedder
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.common.testing.interpret_test import (
    FakeModelForTestingInterpret,
    FakePredictorForTestingInterpret,
)


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

        # This checks for a bug that arose with a change in the pytorch API.  We want to be sure we
        # can handle the case where we have to re-encode a vocab item because we didn't save it in
        # our fake embedding matrix (see Hotflip docstring for more info).
        hotflipper = Hotflip(predictor, max_tokens=50)
        hotflipper.initialize()
        hotflipper._first_order_taylor(
            grad=torch.rand((10,)).numpy(), token_idx=torch.tensor(60), sign=1
        )

    def test_interpret_fails_when_embedding_layer_not_found(self):
        inputs = {"sentence": "I always write unit tests for my code."}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = TextClassifierPredictor(model, TextClassificationJsonReader())

        hotflipper = Hotflip(predictor)
        with raises(RuntimeError):
            hotflipper.initialize()

    def test_interpret_works_with_custom_embedding_layer(self):
        inputs = {"sentence": "I always write unit tests for my code"}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = FakePredictorForTestingInterpret(model, TextClassificationJsonReader())

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
