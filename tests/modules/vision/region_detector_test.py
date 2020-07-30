from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone
from allennlp.modules.vision.region_detector import FasterRcnnRegionDetector


class TestFasterRcnnRegionDetector(AllenNlpTestCase):
    def test_forward_runs(self):
        loader = DetectronImageLoader()
        backbone = ResnetBackbone()
        num_boxes = 40
        detector = FasterRcnnRegionDetector(test_detections_per_image=num_boxes)
        image_pixels, image_size = loader(self.FIXTURES_ROOT / "detectron" / "000000001268.jpg")
        assert image_size[0] == 800
        assert image_size[1] == 1199
        image_pixels = image_pixels.unsqueeze(0).expand(2, -1, -1, -1)
        image_size = image_size.unsqueeze(0).expand(2, -1)
        grid_features = backbone(image_pixels, image_size)
        results = detector(image_pixels, image_size, grid_features)
        assert results["coordinates"].size() == (2, num_boxes, 4)
        assert results["features"].size() == (2, num_boxes, 2048)
        assert results["class_probabilities"].size() == (2, num_boxes, 1600)
