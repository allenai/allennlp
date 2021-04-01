import pytest
import torch
import os

from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase

from transformers import AutoModel, AutoConfig


class TestCachedTransformers(AllenNlpTestCase):
    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            transformer = cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )

    def test_from_pretrained_avoids_weights_download_if_override_weights(self):
        # only download config because downloading pretrained weights in addition takes too long
        transformer = AutoModel.from_config(
            AutoConfig.from_pretrained("bert-base-uncased", cache_dir=self.TEST_DIR)
        )
        # clear cache directory
        for f in os.listdir(str(self.TEST_DIR)):
            os.remove(str(self.TEST_DIR) + "/" + f)
        assert len(os.listdir(str(self.TEST_DIR))) == 0

        save_weights_path = str(self.TEST_DIR) + "/bert_weights.pth"
        torch.save(transformer.state_dict(), save_weights_path)

        pre_download_num_files = len(os.listdir(str(self.TEST_DIR)))
        override_transformer = cached_transformers.get(
            "bert-base-uncased",
            False,
            override_weights_file=save_weights_path,
            cache_dir=self.TEST_DIR,
        )
        post_download_num_files = len(os.listdir(str(self.TEST_DIR)))
        # check that only three files were downloaded (.json, .[etag], .lock), for config.json
        # if more than three files were downloaded, then model weights were also (incorrectly) downloaded
        # NOTE: downloaded files are not explicitly detailed in Huggingface's public API,
        # so this assertion could fail in the future
        assert post_download_num_files - pre_download_num_files == 3

        # check that override weights were loaded correctly
        for p1, p2 in zip(transformer.parameters(), override_transformer.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
