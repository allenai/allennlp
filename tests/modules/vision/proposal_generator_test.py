from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone
from allennlp.modules.vision.proposal_generator import FasterRcnnProposalGenerator


class TestFasterRcnnProposalGenerator(AllenNlpTestCase):
    def test_forward_runs(self):
        loader = DetectronImageLoader()
        backbone = ResnetBackbone()
        n_boxes = 40
        generator = FasterRcnnProposalGenerator(test_detections_per_image=n_boxes)
        image_pixels, image_size = loader(self.FIXTURES_ROOT / "detectron" / "000000001268.jpg")
        assert image_size[0] == 800
        assert image_size[1] == 1199
        image_pixels = image_pixels.unsqueeze(0).expand(2, -1, -1, -1)
        image_size = image_size.unsqueeze(0).expand(2, -1)
        grid_features = backbone(image_pixels, image_size)
        box_features, box_coords, box_cls_probs = generator(image_pixels, image_size, grid_features)
        assert box_coords.size() == (2, n_boxes, 4)
        assert box_features.size() == (2, n_boxes, 2048)
        assert box_cls_probs.size() == (2, n_boxes, 1600)
