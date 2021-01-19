from allennlp.common.lazy import Lazy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Batch, Vocabulary
from allennlp.data.dataset_readers import VisualEntailmentReader
from allennlp.data.image_loader import TorchImageLoader
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.vision.grid_embedder import NullGridEmbedder
from allennlp.modules.vision.region_detector import RandomRegionDetector


class TestVisualEntailmentReader(AllenNlpTestCase):
    def test_read(self):
        reader = VisualEntailmentReader(
            image_dir=self.FIXTURES_ROOT / "data" / "visual_entailment",
            image_loader=TorchImageLoader(),
            image_featurizer=Lazy(NullGridEmbedder),
            region_detector=Lazy(RandomRegionDetector),
            tokenizer=WhitespaceTokenizer(),
            token_indexers={"tokens": SingleIdTokenIndexer()},
        )
        instances = list(reader.read("test_fixtures/data/visual_entailment/sample_pairs.jsonl"))
        assert len(instances) == 16

        instance = instances[0]
        assert len(instance.fields) == 5
        assert len(instance["hypothesis"]) == 4
        sentence_tokens = [t.text for t in instance["hypothesis"]]
        assert sentence_tokens == ["A", "toddler", "sleeps", "outside."]
        assert instance["labels"].label == "contradiction"

        batch = Batch(instances)
        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(["entailment", "contradiction", "neutral"], "labels")
        batch.index_instances(vocab)
        tensors = batch.as_tensor_dict()

        # (batch size, num boxes (fake), num features (fake))
        assert tensors["box_features"].size() == (16, 2, 10)

        # (batch size, num boxes (fake), 4 coords)
        assert tensors["box_coordinates"].size() == (16, 2, 4)

        # (batch_size, num boxes (fake),)
        assert tensors["box_mask"].size() == (16, 2)
