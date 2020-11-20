from allennlp.data.image_loader import ImageLoader


def test_detectron_image_loader():
    image_loader = ImageLoader.by_name("detectron")()
    images, sizes = image_loader(["test_fixtures/detectron/000000001268.jpg"])

    # shape: (number of images, color channels, height, width)
    assert images.shape == (1, 3, 800, 1199)

    # shape: (number of images, 2)
    assert sizes.shape == (1, 2)
    assert sizes[0][0] == 800
    assert sizes[0][1] == 1199

    # We should get the same result, but not batched, if we call with just the filename.
    image, size = image_loader("test_fixtures/detectron/000000001268.jpg")
    assert image.shape == (3, 800, 1199)
    assert size.shape == (2,)
