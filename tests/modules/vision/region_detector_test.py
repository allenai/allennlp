from allennlp.common.detectron import DetectronConfig
from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone
from allennlp.modules.vision.region_detector import FasterRcnnRegionDetector


class TestFasterRcnnRegionDetector(AllenNlpTestCase):
    @requires_gpu
    def test_forward_runs(self):
        num_boxes = 100
        batch_size = 2
        config = DetectronConfig.from_flat_parameters(device=0)

        loader = DetectronImageLoader()
        backbone = ResnetBackbone(config=config)
        detector = FasterRcnnRegionDetector(config=config, detections_per_image=num_boxes)

        image_pixels, image_size = loader(self.FIXTURES_ROOT / "detectron" / "000000001268.jpg")

        assert image_size[0] == 800
        assert image_size[1] == 1199

        image_pixels = image_pixels.unsqueeze(0).expand(batch_size, -1, -1, -1)
        image_size = image_size.unsqueeze(0).expand(batch_size, -1)

        grid_features = backbone(image_pixels, image_size)
        results = detector(image_pixels, image_size, grid_features)

        assert results["coordinates"].size() == (batch_size, num_boxes, 4)
        assert results["features"].size() == (batch_size, num_boxes, 2048)
        assert results["class_probs"].size() == (batch_size, num_boxes, 1600)
        assert results["num_regions"].size() == (batch_size,)

        for batch_index in range(batch_size):
            # There is some non-determinism in detectron in how many regions it detects on this
            # image - most of the time it returns 60, but occasionally returns 64.  This is hedging
            # against the non-determinism, so the test doesn't randomly fail.
            actual_num_regions = results["num_regions"][batch_index].item()
            assert actual_num_regions in {60, 64}
