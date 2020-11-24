import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import VQAv2Reader
from allennlp.data.image_loader import DetectronImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector


class TestVQAv2Reader(AllenNlpTestCase):
    def test_read(self):
        reader = VQAv2Reader(
            image_dir=self.FIXTURES_ROOT / "data" / "vqav2" / "images",
            image_loader=DetectronImageLoader(),
            image_featurizer=NullGridEmbedder(),
            region_detector=RandomRegionDetector(),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("unittest"))
        assert len(instances) == 3

        instance = instances[0]
        assert len(instance.fields) == 5
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
