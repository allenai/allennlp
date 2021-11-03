import pytest
import torch
import torchvision

from allennlp.common.testing import AllenNlpTestCase, multi_device
from allennlp.data.image_loader import TorchImageLoader


class TorchImageLoaderTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.image_fixture_path = str(
            self.FIXTURES_ROOT / "data" / "images" / "COCO_train2014_000000458752.jpg"
        )

        torchvision.set_image_backend("accimage")

        # Create a few small images of different sizes from the fixture.
        image = torchvision.io.read_image(self.image_fixture_path)
        assert image.shape == (3, 480, 640)

        image1 = image[:, 0:7, 0:15]
        image2 = image[:, 0:9, 0:12]
        torchvision.io.write_jpeg(image1, str(self.TEST_DIR / "image1.jpg"))
        torchvision.io.write_jpeg(image2, str(self.TEST_DIR / "image2.jpg"))

    @multi_device
    @pytest.mark.parametrize(
        "loader_params",
        [
            {"size_divisibility": 0, "pad_value": 0.0},
            {"size_divisibility": 1, "pad_value": 0.0},
            {"size_divisibility": 4, "pad_value": 0.0},
        ],
        ids=str,
    )
    def test_basic_load(self, device, loader_params):
        loader = TorchImageLoader(resize=False, normalize=False, device=device, **loader_params)
        torch_device = torch.device(device)
        images, sizes = loader([self.TEST_DIR / "image1.jpg", self.TEST_DIR / "image2.jpg"])
        assert images.device == torch_device
        assert sizes.device == torch_device
        assert images.shape[0] == 2
        assert images.shape[1] == 3
        assert sizes.shape == (2, 2)
        assert list(sizes[0]) == [7, 15]
        assert list(sizes[1]) == [9, 12]
        if loader.size_divisibility <= 1:
            assert images.shape[2] == 9
            assert images.shape[3] == 15
        else:
            assert images.shape[2] >= 9
            assert images.shape[3] >= 15
            assert (images.shape[2] / loader.size_divisibility) % 1 == 0

        image, size = loader(self.TEST_DIR / "image1.jpg")
        assert image.device == torch_device
        assert size.device == torch_device
        assert len(image.shape) == 3
        assert list(size) == [7, 15]

    @multi_device
    def test_resize_and_normalize(self, device):
        loader = TorchImageLoader(resize=True, normalize=True, device=device)
        torch_device = torch.device(device)
        image, size = loader(self.image_fixture_path)
        assert image.device == torch_device
        assert size.device == torch_device
        assert image.shape[1] == 800

    def test_resize_and_normalize_matches_generalized_rcnn_transform(self):
        loader = TorchImageLoader(resize=True, normalize=True, size_divisibility=32)
        transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(
            loader.min_size, loader.max_size, loader.pixel_mean, loader.pixel_std
        )

        loaded_image, _ = loader([self.image_fixture_path])

        raw_image, _ = TorchImageLoader(resize=False, normalize=False)(self.image_fixture_path)
        transformed_raw_image, _ = transform([raw_image])

        assert loaded_image.shape == transformed_raw_image.tensors.shape
