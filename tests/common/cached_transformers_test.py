import pytest
import torch
import os

from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase


class TestCachedTransformers(AllenNlpTestCase):
    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            transformer = cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
            assert os.path.isfile(self.TEST_DIR + "/" + "bert-base-uncased.bin")
            assert os.path.isfile(self.TEST_DIR + "/" + "config.json")
            os.remove(self.TEST_DIR + "/" + "bert-base-uncased.bin")
            os.remove(self.TEST_DIR + "/" + "config.json")

            torch.save(transformer.module.state_dict(), self.TEST_DIR + "/" + "bert_weights.pth")
            cached_transformers.get(
                "bert-base-uncased",
                True,
                override_weights_file=self.TEST_DIR + "/" + "bert_weights.pth",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
            assert not os.path.isfile(self.TEST_DIR + "/" + "bert-base-uncased.bin")
            assert os.path.isfile(self.TEST_DIR + "/" + "config.json")

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
