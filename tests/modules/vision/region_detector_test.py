import torchvision

from allennlp.common.testing import AllenNlpTestCase, requires_gpu
from allennlp.data.image_loader import TorchImageLoader
from allennlp.modules.vision.grid_embedder import ResnetBackbone
from allennlp.modules.vision.region_detector import FasterRcnnRegionDetector


class TestFasterRcnnRegionDetector(AllenNlpTestCase):
    @requires_gpu
    def test_forward_runs(self):
        loader = TorchImageLoader(resize=True, normalize=True, device="cuda:0")
        backbone = ResnetBackbone().to(device="cuda:0")
        backbone.eval()
        detector = FasterRcnnRegionDetector().to(device="cuda:0")
        detector.eval()

        image_path = self.FIXTURES_ROOT / "data" / "images" / "COCO_train2014_000000458752.jpg"

        images, sizes = loader([image_path, image_path])
        image_features = backbone(images, sizes)
        del backbone
        detections = detector(images, sizes, image_features)
        del detector

        assert len(detections.features) == 2
        assert len(detections.boxes) == 2
        assert len(detections.class_probs) == 2
        assert len(detections.class_labels) == 2

        assert detections.features[0].shape[0] >= 1
        assert detections.features[0].shape[1] == 1024
        assert (
            detections.features[0].shape[0]
            == detections.boxes[0].shape[0]
            == detections.class_probs[0].shape[0]
            == detections.class_labels[0].shape[0]
        )

        # Okay, cool, so far so good. Now let's make sure the output we got
        # actually matches exactly what we would get using the full pipeline
        # directly from torchvision.
        raw_loader = TorchImageLoader(resize=False, normalize=False, device="cuda:0")
        image, _ = raw_loader(image_path)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to("cuda:0")
        model.eval()
        result = model([image, image])
        # We can't compare the boxes directly because the boxes here are post-processed
        # back to reference the original un-resized image. But we can compare
        # the labels and scores. They should match exactly.
        assert (result[0]["labels"] == detections.class_labels[0]).all()
        assert (result[0]["scores"] == detections.class_probs[0]).all()
