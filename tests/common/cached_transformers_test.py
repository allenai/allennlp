import pytest

from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase


class TestCachedTransformers(AllenNlpTestCase):
    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
