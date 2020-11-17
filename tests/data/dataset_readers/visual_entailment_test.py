from typing import Dict

import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import VisualEntailmentReader
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
    ) -> Dict[str, torch.FloatTensor]:
        self.calls += 1
        batch_size, num_features, height, width = raw_images.size()
        features = torch.ones(batch_size, 1, 10, dtype=featurized_images.dtype)
        coordinates = torch.zeros(batch_size, 1, 4, dtype=image_sizes.dtype)
        for image_num in range(batch_size):
            coordinates[image_num, 0, 2] = image_sizes[image_num, 0]
            coordinates[image_num, 0, 3] = image_sizes[image_num, 1]
        return {"features": features, "coordinates": coordinates}


class TestVisualEntailmentReader(AllenNlpTestCase):
    def test_read(self):
        detector = FakeRegionDetector()
        reader = VisualEntailmentReader(
            image_dir=self.FIXTURES_ROOT / "data" / "visual_entailment",
            image_loader=DetectronImageLoader(),
            image_featurizer=NullGridEmbedder(),
            region_detector=detector,
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("test_fixtures/data/visual_entailment/sample_pairs.jsonl"))
        assert len(instances) == 16

        instance = instances[0]
        assert len(instance.fields) == 5
        assert len(instance["sentence1"]) == 12
        assert len(instance["sentence2"]) == 4
        sentence_tokens = [t.text for t in instance["sentence1"]]
        assert sentence_tokens[:6] == ["A", "toddler", "poses", "in", "front", "of"]
        assert instance["label"].label == "contradiction"

        batch = Batch(instances)
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["entailment", "contradiction", "neutral"], "labels")
        batch.index_instances(vocab)
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (16, 1, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (16, 1, 4)

        # We have 2 images total, and 16 instances.  Those 2 images are processed two at a time in
        # the region detector, and the results are cached, so we should only see the region detector
        # called 1 time with this data.  This is testing the feature caching functionality in the
        # dataset reader.
        assert detector.calls == 1
