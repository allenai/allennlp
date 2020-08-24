import os
import glob
import argparse
from tqdm import tqdm

import torch

from allennlp.common.file_utils import TensorCache, cached_path
from allennlp.data import DetectronImageLoader
from allennlp.modules.vision import ResnetBackbone, FasterRcnnRegionDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--use-cuda", action="store_true", help="use GPU if one is available")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.cache_path, exist_ok=True)
    features_cache = TensorCache(os.path.join(args.cache_path, "features"))
    coordinates_cache = TensorCache(os.path.join(args.cache_path, "coordinates"))
    image_paths = list(glob.iglob(os.path.join(args.image_dir, "**", "*.png"), recursive=True))
    image_keys = [os.path.basename(filename) for filename in image_paths]

    image_loader = DetectronImageLoader()
    image_featurizer = ResnetBackbone()
    region_detector = FasterRcnnRegionDetector()
    if torch.cuda.is_available() and args.use_cuda:
        image_featurizer.cuda()
        region_detector.cuda()

    images, sizes = image_loader(image_paths)
    for index in tqdm(range(0, images.shape[0], args.batch_size)):
        end = min(index+args.batch_size, images.shape[0])
        batch_images = images[index:end].to(image_featurizer.device)
        batch_shapes = sizes[index:end]
        with torch.no_grad():
            featurized_images = image_featurizer(batch_images)
            detector_results = region_detector(batch_images, batch_shapes, featurized_images)
        features = detector_results["features"]
        coordinates = detector_results["coordinates"]
        for subindex in range(index, end):
            features_cache[image_keys[subindex]] = features[subindex-index]
            coordinates_cache[image_keys[subindex]] = coordinates[subindex-index]
