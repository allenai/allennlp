from typing import Dict

import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import Nlvr2Reader
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector


class TestNlvr2Reader(AllenNlpTestCase):
    def test_read(self):
        detector = RandomRegionDetector()
        reader = Nlvr2Reader(
            image_dir=self.FIXTURES_ROOT / "data" / "nlvr2",
            image_loader=DetectronImageLoader(),
            image_featurizer=NullGridEmbedder(),
            region_detector=detector,
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("test_fixtures/data/nlvr2/tiny-dev.json"))
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

        # We have 8 images total, and 8 instances.  Those 8 images are processed two at a time in
        # the region detector, and the results are cached, so we should only see the region detector
        # called 4 times with this data.  This is testing the feature caching functionality in the
        # dataset reader.
        assert detector.calls == 4
