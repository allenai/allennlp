import os
import glob
import argparse
from tqdm import tqdm

import torch
from torchvision.datasets.folder import IMG_EXTENSIONS

from allennlp.common.file_utils import TensorCache, cached_path
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
    features_cache = TensorCache(os.path.join(args.cache_dir, "features"))
    coordinates_cache = TensorCache(os.path.join(args.cache_dir, "coordinates"))
    image_paths = []
    for extension in IMG_EXTENSIONS:
        image_paths += list(glob.iglob(os.path.join(args.image_dir, "**", "*."+extension), recursive=True))
    filtered_image_paths = []
    image_keys = []
    for path in image_paths:
        key = os.path.basename(path)
        if key in features_cache and key in coordinates_cache:
            continue
        image_keys.append(key)
        filtered_image_paths.append(path)
    image_paths = filtered_image_paths

    image_loader = DetectronImageLoader()
    image_featurizer = ResnetBackbone()
    region_detector = FasterRcnnRegionDetector()
    if torch.cuda.is_available() and args.use_cuda:
        image_featurizer.cuda()
        region_detector.cuda()

    for index in tqdm(range(0, len(image_paths), args.batch_size)):
        end = min(index+args.batch_size, len(image_paths))
        batch_images, batch_shapes = image_loader(image_paths[index:end])
        with torch.no_grad():
            featurized_images = image_featurizer(batch_images, batch_shapes)
            detector_results = region_detector(batch_images, batch_shapes, featurized_images)
            features = detector_results["features"]
            coordinates = detector_results["coordinates"]
        for subindex in range(index, end):
            features_cache[image_keys[subindex]] = features[subindex-index].cpu()
            coordinates_cache[image_keys[subindex]] = coordinates[subindex-index].cpu()
