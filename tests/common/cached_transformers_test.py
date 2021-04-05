import pytest
import torch
import os
import json

from allennlp.common import cached_transformers
from allennlp.common.testing import AllenNlpTestCase

from transformers import AutoModel, AutoConfig


class TestCachedTransformers(AllenNlpTestCase):
    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )

    def test_from_pretrained_avoids_weights_download_if_override_weights(self):
        # only download config because downloading pretrained weights in addition takes too long
        transformer = AutoModel.from_config(
            AutoConfig.from_pretrained("epwalsh/bert-xsmall-dummy", cache_dir=self.TEST_DIR)
        )
        # clear cache directory
        for f in os.listdir(str(self.TEST_DIR)):
            os.remove(str(self.TEST_DIR) + "/" + f)
        assert len(os.listdir(str(self.TEST_DIR))) == 0

        save_weights_path = str(self.TEST_DIR) + "/bert_weights.pth"
        torch.save(transformer.state_dict(), save_weights_path)

        override_transformer = cached_transformers.get(
            "bert-base-uncased",
            False,
            override_weights_file=save_weights_path,
            cache_dir=self.TEST_DIR,
        )
        # check that only three files were downloaded (filename.json, filename, filename.lock), for config.json
        # if more than three files were downloaded, then model weights were also (incorrectly) downloaded
        # NOTE: downloaded files are not explicitly detailed in Huggingface's public API,
        # so this assertion could fail in the future
        json_fnames = [fname for fname in os.listdir(str(self.TEST_DIR)) if fname.endswith(".json")]
        assert len(json_fnames) == 1
        json_data = json.load(open(str(self.TEST_DIR) + "/" + json_fnames[0]))
        assert (
            json_data["url"] == "https://huggingface.co/bert-base-uncased/resolve/main/config.json"
        )
        resource_id = os.path.splitext(json_fnames[0])[0]
        assert set(os.listdir(str(self.TEST_DIR))) == set(
            [json_fnames[0], resource_id, resource_id + ".lock", "bert_weights.pth"]
        )

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
