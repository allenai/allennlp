import math

from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone


class TestResnetBackbone(AllenNlpTestCase):
    @requires_gpu
    def test_forward_runs(self):
        loader = DetectronImageLoader()
        backbone = ResnetBackbone(device=0)
        image_pixels, image_size = loader(self.FIXTURES_ROOT / "detectron" / "000000001268.jpg")
        image_height = image_size[0]  # 800 for the image above
        image_width = image_size[1]  # 1199 for the image above
        result = backbone(image_pixels.unsqueeze(0), image_size.unsqueeze(0))

        # Stride is currently 16 and output dim is 1024 in the default backbone; just FYI.
        expected_height = math.ceil(float(image_height) / backbone.get_stride())  # 50
        expected_width = math.ceil(float(image_width) / backbone.get_stride())  # 75
        assert result.size() == (1, backbone.get_output_dim(), expected_height, expected_width)

    def test_output_shape_is_correct(self):
        backbone = ResnetBackbone()
        assert backbone.get_output_dim() == 1024
        assert backbone.get_stride() == 16
