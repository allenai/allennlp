from pytest import approx
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters import IntegratedGradient


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
