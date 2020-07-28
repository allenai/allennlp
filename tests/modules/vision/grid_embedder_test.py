from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.vision.grid_embedder import ResnetBackbone


class TestResnetBackbone(AllenNlpTestCase):
    def test_output_shape_is_correct(self):
        backbone = ResnetBackbone()
        print(backbone)
        assert backbone.get_output_dim() == 1024
        assert backbone.get_stride() == 16
