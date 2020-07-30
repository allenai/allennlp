from typing import Dict

import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import Nlvr2Reader
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


class FakeGridEmbedder(GridEmbedder):
    def forward(self, images: torch.FloatTensor):
        return images


class FakeRegionDetector(RegionDetector):
    def forward(
        self,
        raw_images: torch.FloatTensor,
        image_sizes: torch.IntTensor,
        featurized_images: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        batch_size, num_features, height, width = raw_images.size()
        features = torch.ones(batch_size, 1, 10, dtype=featurized_images.dtype)
        coordinates = torch.zeros(batch_size, 1, 4, dtype=image_sizes.dtype)
        for image_num in range(batch_size):
            coordinates[image_num, 0, 2] = image_sizes[image_num, 0]
            coordinates[image_num, 0, 3] = image_sizes[image_num, 1]
        return {"features": features, "coordinates": coordinates}


class TestNlvr2Reader(AllenNlpTestCase):
    def test_read(self):
        reader = Nlvr2Reader(
            image_dir=self.FIXTURES_ROOT / "data" / "nlvr2",
            image_loader=DetectronImageLoader(),
            image_featurizer=FakeGridEmbedder(),
            region_detector=FakeRegionDetector(),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = reader.read("test_fixtures/data/nlvr2/tiny-dev.json")
        assert len(instances) == 8

        instance = instances[0]
        assert len(instance.fields) == 5
        assert len(instance["sentence"]) == 18
        sentence_tokens = [t.text for t in instance["sentence"]]
        assert sentence_tokens[:6] == ["The", "right", "image", "shows", "a", "curving"]
        assert instance["label"].label == 1
        assert instance["identifier"].metadata == "dev-850-0-0"

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, 2 images per instance, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (8, 2, 1, 10)

        # (batch size, 2 images per instance, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (8, 2, 1, 4)
