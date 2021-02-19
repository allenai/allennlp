import pytest

from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase


class TestCachedTransformers(AllenNlpTestCase):
    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises(OSError) as execinfo:
            cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
        assert "Can't load config for 'bert-base-uncased'" in str(execinfo.value)

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        with pytest.raises(OSError) as execinfo:
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
        assert "Can't load config for 'bert-base-uncased'" in str(execinfo.value)
