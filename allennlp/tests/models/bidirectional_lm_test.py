# pylint: disable=invalid-name,arguments-differ,abstract-method
from typing import Dict, Iterable

import numpy as np

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer

@DatasetReader.register("test-sentence")
class _SentenceDatasetReader(DatasetReader):
    """
    Simple dataset reader that just reads sentences from a file and tokenizes them.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: WordTokenizer = WordTokenizer()) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers

    def _read(self, filename: str) -> Iterable[Instance]:
        with open(filename) as f:
            for line in f:
                tokens = self.tokenizer.tokenize(line)
                field = TextField(tokens, self.token_indexers)
                yield Instance({"tokens": field})


class TestBidirectionalLM(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'bidirectional_lm' / 'experiment.jsonnet',
                          self.FIXTURES_ROOT / 'bidirectional_lm' / 'sentences.txt')

    def test_bidirectional_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        result = self.model(**training_tensors)

        assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings", "mask"}

        # The model should have removed the BOS / EOS tokens.
        embeddings = result["lm_embeddings"]
        assert tuple(embeddings.shape) == (2, 6, 14)

        loss = result["loss"].item()
        forward_loss = result["forward_loss"].item()
        backward_loss = result["backward_loss"].item()

        np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)
