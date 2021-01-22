import os
import glob
import argparse
from os import PathLike
from typing import Union, List

from tqdm import tqdm

import torch
from torchvision.datasets.folder import IMG_EXTENSIONS

from allennlp.common.file_utils import TensorCache
from allennlp.data import DetectronImageLoader
from allennlp.modules.vision import ResnetBackbone, FasterRcnnRegionDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--use-cuda", action="store_true", help="use GPU if one is available")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    feature_cache = TensorCache(os.path.join(args.cache_dir, "features"))
    coordinates_cache = TensorCache(os.path.join(args.cache_dir, "coordinates"))
    image_paths = []
    for extension in IMG_EXTENSIONS:
        extension = extension.lstrip(
            "."
        )  # Some versions of detectron have the period. Others do not.
        image_paths += list(
            glob.iglob(os.path.join(args.image_dir, "**", "*." + extension), recursive=True)
        )

    image_loader = DetectronImageLoader()
    image_featurizer = ResnetBackbone()
    region_detector = FasterRcnnRegionDetector()
    if torch.cuda.is_available() and args.use_cuda:
        image_featurizer.cuda()
        region_detector.cuda()

    def process_batch(batch: List[Union[str, PathLike]]):
        batch_images, batch_shapes = image_loader(batch)
        with torch.no_grad():
            featurized_images = image_featurizer(batch_images, batch_shapes)
            detector_results = region_detector(batch_images, batch_shapes, featurized_images)
            features = detector_results["features"]
            coordinates = detector_results["coordinates"]
        for filename, image_features, image_coordinates in zip(batch, features, coordinates):
            filename = os.path.basename(filename)
            feature_cache[filename] = features.cpu()
            coordinates_cache[filename] = coordinates.cpu()

    image_path_batch = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        key = os.path.basename(image_path)
        if key in feature_cache and key in coordinates_cache:
            continue
        image_path_batch.append(image_path)

        if len(image_path_batch) >= args.batch_size:
            process_batch(image_path_batch)
            image_path_batch.clear()
    if len(image_path_batch) > 0:
        process_batch(image_path_batch)
