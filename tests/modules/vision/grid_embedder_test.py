from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.data.image_loader import TorchImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone


class TestResnetBackbone(AllenNlpTestCase):
    @requires_gpu
    def test_forward_runs(self):
        loader = TorchImageLoader(device="cuda:0")
        backbone = ResnetBackbone().to("cuda:0")

        image_pixels, image_size = loader(
            [self.FIXTURES_ROOT / "data" / "images" / "COCO_train2014_000000458752.jpg"]
        )
        result = backbone(image_pixels, image_size)
        assert tuple(result.keys()) == backbone.get_feature_names()
