import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.lazy import Lazy
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import VQAv2Reader
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector


class TestVQAv2Reader(AllenNlpTestCase):
    def test_read(self):
        reader = VQAv2Reader(
            image_dir=self.FIXTURES_ROOT / "data" / "vqav2" / "images",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("unittest"))
        assert len(instances) == 3

        instance = instances[0]
        assert len(instance.fields) == 6
        assert len(instance["question"]) == 7
        question_tokens = [t.text for t in instance["question"]]
        assert question_tokens == ["What", "is", "this", "photo", "taken", "looking", "through?"]
        assert len(instance["labels"]) == 5
        labels = [field.label for field in instance["labels"].field_list]
        assert labels == ["net", "netting", "mesh", "pitcher", "orange"]
        assert torch.allclose(
            instance["label_weights"].tensor,
            torch.tensor([1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3]),
        )

        batch = Batch(instances)
        batch.index_instances(Vocabulary())
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (3, 2, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (3, 2, 4)

        # (batch size, num boxes (fake),)
        assert tensors["box_mask"].size() == (3, 2)

        # Nothing should be masked out since the number of fake boxes is the same
        # for each item in the batch.
        assert tensors["box_mask"].all()

    def test_read_without_images(self):
        reader = VQAv2Reader(
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("unittest"))
        assert len(instances) == 3
        assert "box_coordinates" not in instances[0]
        assert "box_features" not in instances[0]
        assert "box_mask" not in instances[0]
