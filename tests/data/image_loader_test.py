import pytest
import torchvision

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.image_loader import TorchImageLoader


@pytest.fixture(
    params=[
        {"size_divisibility": 0, "pad_value": 0.0},
        {"size_divisibility": 1, "pad_value": 0.0},
        {"size_divisibility": 4, "pad_value": 0.0},
    ],
    ids=str,
)
def loader(request):
    return TorchImageLoader(**request.param)


class TorchImageLoaderTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        image_fixture_path = str(
            self.FIXTURES_ROOT
            / "data"
            / "vqav2"
            / "images"
            / "test_fixture"
            / "COCO_train2014_000000458752.jpg"
        )

        # Create a few small images of different sizes from the fixture.
        image = torchvision.io.read_image(image_fixture_path)
        assert image.shape == (3, 480, 640)

        image1 = image[:, 0:7, 0:15]
        image2 = image[:, 0:9, 0:12]
        torchvision.io.write_jpeg(image1, str(self.TEST_DIR / "image1.jpg"))
        torchvision.io.write_jpeg(image2, str(self.TEST_DIR / "image2.jpg"))

    def test_load(self, loader):
        images, sizes = loader([self.TEST_DIR / "image1.jpg", self.TEST_DIR / "image2.jpg"])
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
        assert len(image.shape) == 3
        assert list(size) == [7, 15]
