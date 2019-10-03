from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.saliency_interpreters import SmoothGradient


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
