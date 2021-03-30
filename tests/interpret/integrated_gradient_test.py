from pytest import approx, raises
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.interpret.saliency_interpreters import IntegratedGradient

from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.common.testing.interpret_test import (
    FakeModelForTestingInterpret,
    FakePredictorForTestingInterpret,
)


class TestIntegratedGradient(AllenNlpTestCase):
    def test_integrated_gradient(self):
        inputs = {"sentence": "It was the ending that I hated"}
        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        predictor = Predictor.from_archive(archive, "text_classifier")

        interpreter = IntegratedGradient(predictor)
        interpretation = interpreter.saliency_interpret_from_json(inputs)
        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        grad_input_1 = interpretation["instance_1"]["grad_input_1"]
        assert len(grad_input_1) == 7  # 7 words in input

        # two interpretations should be identical for integrated gradients
        repeat_interpretation = interpreter.saliency_interpret_from_json(inputs)
        repeat_grad_input_1 = repeat_interpretation["instance_1"]["grad_input_1"]
        for grad, repeat_grad in zip(grad_input_1, repeat_grad_input_1):
            assert grad == approx(repeat_grad)

    def test_interpret_fails_when_embedding_layer_not_found(self):
        inputs = {"sentence": "It was the ending that I hated"}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = TextClassifierPredictor(model, TextClassificationJsonReader())

        interpreter = IntegratedGradient(predictor)
        with raises(RuntimeError):
            interpreter.saliency_interpret_from_json(inputs)

    def test_interpret_works_with_custom_embedding_layer(self):
        inputs = {"sentence": "It was the ending that I hated"}
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace([w for w in inputs["sentence"].split(" ")])
        model = FakeModelForTestingInterpret(vocab, max_tokens=len(inputs["sentence"].split(" ")))
        predictor = FakePredictorForTestingInterpret(model, TextClassificationJsonReader())
        interpreter = IntegratedGradient(predictor)

        interpretation = interpreter.saliency_interpret_from_json(inputs)

        assert interpretation is not None
        assert "instance_1" in interpretation
        assert "grad_input_1" in interpretation["instance_1"]
        grad_input_1 = interpretation["instance_1"]["grad_input_1"]
        assert len(grad_input_1) == 7  # 7 words in input
