from allennlp.common.testing import AllenNlpTestCase


class DetectronUtilsTest(AllenNlpTestCase):
    def test_detectron_processor(self):
        from allennlp.data.dataset_readers.dataset_utils.detectron_utils import DetectronProcessor
        from allennlp.common.detectron import get_detectron_cfg

        cfg = get_detectron_cfg("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        processor = DetectronProcessor(cfg)
        processed_images = processor(
            [
                self.FIXTURES_ROOT / "detectron" / f
                for f in ["000000001268.jpg", "000000003156.jpg", "000000008211.jpg"]
            ]
        )
        assert len(processed_images) == 3
        for processed_image in processed_images:
            assert "instances/pred_boxes" in processed_image
            assert "instances/pred_classes" in processed_image
            assert "instances/pred_masks" in processed_image
            assert "instances/scores" in processed_image
            # Assert that all fields have the same length.
            assert len({len(field) for field in processed_image.values()}) == 1
