from typing import Dict

import torch
from torch import Tensor

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import GQAReader
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


class FakeRegionDetector(RegionDetector):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        raw_images: torch.FloatTensor,
        image_sizes: torch.IntTensor,
        featurized_images: torch.FloatTensor,
    ) -> Dict[str, Tensor]:
        self.calls += 1
        batch_size, num_features, height, width = raw_images.size()
        features = torch.ones(batch_size, 1, 10, dtype=featurized_images.dtype)
        coordinates = torch.zeros(batch_size, 1, 4, dtype=image_sizes.dtype)
        for image_num in range(batch_size):
            coordinates[image_num, 0, 2] = image_sizes[image_num, 0]
            coordinates[image_num, 0, 3] = image_sizes[image_num, 1]
        return {"features": features, "coordinates": coordinates}


class TestGQAReader(AllenNlpTestCase):
    def test_read(self):
        reader = GQAReader(
            image_dir=self.FIXTURES_ROOT / "data" / "gqa" / "images",
            image_loader=DetectronImageLoader(),
            image_featurizer=NullGridEmbedder(),
            region_detector=FakeRegionDetector(),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )

        instances = list(reader.read("test_fixtures/data/gqa/questions.json"))
        assert len(instances) == 1

        instance = instances[0]
        assert len(instance.fields) == 4
        assert len(instance["question"]) == 6
        question_tokens = [t.text for t in instance["question"]]
        assert question_tokens == ["What", "is", "hanging", "above", "the", "chalkboard?"]
        assert instance["label"].label == "picture"

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (1, 1, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (1, 1, 4)

    def test_read_from_dir(self):
        reader = GQAReader(
            image_dir=self.FIXTURES_ROOT / "data" / "gqa" / "images",
            image_loader=DetectronImageLoader(),
            image_featurizer=NullGridEmbedder(),
            region_detector=FakeRegionDetector(),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        # Test reading from multiple files in a directory
        instances = list(reader.read("test_fixtures/data/gqa/question_dir/"))
        assert len(instances) == 2

        instance = instances[1]
        assert len(instance.fields) == 4
        assert len(instance["question"]) == 10
        question_tokens = [t.text for t in instance["question"]]
        assert question_tokens == [
            "Does",
            "the",
            "table",
            "below",
            "the",
            "water",
            "look",
            "wooden",
            "and",
            "round?",
        ]
        assert instance["label"].label == "yes"

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (2, 1, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (2, 1, 4)
