from pytest import raises
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.interpret.saliency_interpreters import SmoothGradient

from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.common.testing.interpret_test import (
    FakeModelForTestingInterpret,
    FakePredictorForTestingInterpret,
)


class TestSmoothGradient(AllenNlpTestCase):
    def test_smooth_gradient(self):
        inputs = {"sentence": "It was the ending that I hated"}
        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "text_classifier")

        interpreter = SmoothGradient(predictor)
        interpretation = interpreter.saliency_interpret_from_json(inputs)
        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        assert len(interpretation["instance_1"]["grad_input_1"]) == 7  # 7 words in input

    def test_interpret_fails_when_embedding_layer_not_found(self):
        inputs = {"sentence": "It was the ending that I hated"}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = TextClassifierPredictor(model, TextClassificationJsonReader())

        interpreter = SmoothGradient(predictor)
        with raises(RuntimeError):
            interpreter.saliency_interpret_from_json(inputs)

    def test_interpret_works_with_custom_embedding_layer(self):
        inputs = {"sentence": "It was the ending that I hated"}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = FakePredictorForTestingInterpret(model, TextClassificationJsonReader())
        interpreter = SmoothGradient(predictor)

        interpretation = interpreter.saliency_interpret_from_json(inputs)

        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        grad_input_1 = interpretation["instance_1"]["grad_input_1"]
        assert len(grad_input_1) == 7  # 7 words in input
