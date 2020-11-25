from os import PathLike
from typing import Union, Sequence, Tuple, List, cast

from overrides import overrides
import torch
import torchvision
from torch import FloatTensor, IntTensor

from allennlp.common.file_utils import cached_path
from allennlp.common.registrable import Registrable

OnePath = Union[str, PathLike]
ManyPaths = Sequence[OnePath]
ImagesWithSize = Tuple[FloatTensor, IntTensor]


class ImageLoader(Registrable):
    """
    An `ImageLoader` is a callable that takes as input one or more filenames, and outputs two
    tensors: one representing the images themselves, and one that just holds the sizes
    of each image.

    The first tensor is the images and is of shape `(batch_size, color_channels, height, width)`.
    The second tensor is the sizes and is of shape `(batch_size, 2)`, where
    the last dimension contains the height and width, respectively.

    If only a single image is passed (as a `Path` or `str`, instead of a list) then
    the batch dimension will be removed.

    Subclasses only need to implement the `load()` method, which should load a single image
    from a path.

    # Parameters

    size_divisibility : `int`, optional (default = `0`)
        If set to a positive number, padding will be added so that the height
        and width dimensions are divisible by `size_divisibility`.
        Certain models may require this.

    pad_value : `float`, optional (default = `0.0`)
        The value to use for padding.
    """

    default_implementation = "torch"

    def __init__(self, size_divisibility: int = 0, pad_value: float = 0.0) -> None:
        self.size_divisibility = size_divisibility
        self.pad_value = pad_value

    def __call__(self, filename_or_filenames: Union[OnePath, ManyPaths]) -> ImagesWithSize:
        if not isinstance(filename_or_filenames, (list, tuple)):
            image, size = self([filename_or_filenames])  # type: ignore[list-item]
            return cast(FloatTensor, image.squeeze(0)), cast(IntTensor, size.squeeze(0))

        images: List[FloatTensor] = []
        sizes: List[IntTensor] = []
        for filename in filename_or_filenames:
            image = self.load(cached_path(filename))
            size = cast(
                IntTensor, torch.tensor([image.shape[-2], image.shape[-1]], dtype=torch.int32)
            )
            images.append(image)
            sizes.append(size)
        return self._pack_image_list(images, sizes)

    def load(self, filename: OnePath) -> FloatTensor:
        raise NotImplementedError()

    def _pack_image_list(
        self,
        images: List[FloatTensor],
        sizes: List[IntTensor],
    ) -> ImagesWithSize:
        """
        A helper method that subclasses can use to turn a list of individual images into a padded
        batch.
        """
        # Adapted from
        # https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/image_list.py.

        # shape: (batch_size, 2)
        size_tensor = torch.stack(sizes)  # type: ignore[arg-type]

        # shape: (2,)
        max_size = size_tensor.max(0).values

        if self.size_divisibility > 1:
            # shape: (2,)
            max_size = (
                (max_size + self.size_divisibility - 1) // self.size_divisibility
            ) * self.size_divisibility

        batched_shape = [len(images)] + list(images[0].shape[:-2]) + list(max_size)

        # shape: (batch_size, color_channels, max_height, max_width)
        batched_images = images[0].new_full(batched_shape, self.pad_value)

        for image, batch_slice, size in zip(images, batched_images, size_tensor):
            batch_slice[..., : image.shape[-2], : image.shape[-1]].copy_(image)

        return cast(FloatTensor, batched_images), cast(IntTensor, size_tensor)


@ImageLoader.register("torch")
class TorchImageLoader(ImageLoader):
    """
    This is just a wrapper around the default image loader from [torchvision]
    (https://pytorch.org/docs/stable/torchvision/io.html#image).

    # Parameters

    image_backend : `Optional[str]`, optional (default = `None`)
        Set the image backend. Can be one of `"PIL"` or `"accimage"`.
    """

    def __init__(self, image_backend: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if image_backend is not None:
            torchvision.set_image_backend(image_backend)

    @overrides
    def load(self, filename: OnePath) -> FloatTensor:
        return torchvision.io.read_image(filename).float() / 256
