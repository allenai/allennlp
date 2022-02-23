import json
import os

import pytest
import torch
from allennlp.common import cached_transformers
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from transformers import AutoConfig, AutoModel


class TestCachedTransformers(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        cached_transformers._clear_caches()

    def teardown_method(self):
        super().teardown_method()
        cached_transformers._clear_caches()

    def test_get_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get(
                "bert-base-uncased",
                True,
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )

    def clear_test_dir(self):
        for f in os.listdir(str(self.TEST_DIR)):
            os.remove(str(self.TEST_DIR) + "/" + f)
        assert len(os.listdir(str(self.TEST_DIR))) == 0

    def test_from_pretrained_avoids_weights_download_if_override_weights(self):
        config = AutoConfig.from_pretrained("epwalsh/bert-xsmall-dummy", cache_dir=self.TEST_DIR)
        # only download config because downloading pretrained weights in addition takes too long
        transformer = AutoModel.from_config(
            AutoConfig.from_pretrained("epwalsh/bert-xsmall-dummy", cache_dir=self.TEST_DIR)
        )
        transformer = AutoModel.from_config(config)

        # clear cache directory
        self.clear_test_dir()

        save_weights_path = str(self.TEST_DIR / "bert_weights.pth")
        torch.save(transformer.state_dict(), save_weights_path)

        override_transformer = cached_transformers.get(
            "epwalsh/bert-xsmall-dummy",
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
        json_data = json.load(open(str(self.TEST_DIR / json_fnames[0])))
        assert (
            json_data["url"]
            == "https://huggingface.co/epwalsh/bert-xsmall-dummy/resolve/main/config.json"
        )
        resource_id = os.path.splitext(json_fnames[0])[0]
        assert set(os.listdir(str(self.TEST_DIR))) == set(
            [json_fnames[0], resource_id, resource_id + ".lock", "bert_weights.pth"]
        )

        # check that override weights were loaded correctly
        for p1, p2 in zip(transformer.parameters(), override_transformer.parameters()):
            assert p1.data.ne(p2.data).sum() == 0

    def test_reinit_modules_no_op(self):
        # Test the case where reinit_modules is None (default)
        preinit_weights = torch.cat(
            [
                # Comparing all weights of the model is rather complicated, so arbitrarily
                # compare the weights of attention module.
                layer.attention.output.dense.weight
                for layer in cached_transformers.get("bert-base-cased", True).encoder.layer
            ]
        )
        postinit_weights = torch.cat(
            [
                layer.attention.output.dense.weight
                for layer in cached_transformers.get("bert-base-cased", True).encoder.layer
            ]
        )
        assert torch.equal(postinit_weights, preinit_weights)

    def test_reinit_modules_with_layer_indices(self):
        # Comparing all weights of the model is rather complicated, so arbitrarily compare the
        # weights of attention module.
        preinit_weights = torch.cat(
            [
                layer.attention.output.dense.weight
                for layer in cached_transformers.get("bert-base-cased", True).encoder.layer
            ]
        )

        # Test the case when reinit_modules is a valid int.
        postinit_weights = torch.cat(
            [
                layer.attention.output.dense.weight
                for layer in cached_transformers.get(
                    "bert-base-cased", True, reinit_modules=2
                ).encoder.layer
            ]
        )
        assert torch.equal(postinit_weights[:10], preinit_weights[:10])
        assert not torch.equal(postinit_weights[10:], preinit_weights[10:])

        # Test the case when reinit_modules is a valid list of integers.
        postinit_weights = torch.cat(
            [
                layer.attention.output.dense.weight
                for layer in cached_transformers.get(
                    "bert-base-cased", True, reinit_modules=(10, 11)
                ).encoder.layer
            ]
        )
        assert torch.equal(postinit_weights[:10], preinit_weights[:10])
        assert not torch.equal(postinit_weights[10:], preinit_weights[10:])

        # Should raise a ValueError because reinit_modules contains at least one index that is
        # greater than the models maximum number of layers
        with pytest.raises(ValueError):
            _ = cached_transformers.get("bert-base-cased", True, reinit_modules=1000)
        with pytest.raises(ValueError):
            _ = cached_transformers.get("bert-base-cased", True, reinit_modules=(1, 1000))
        # The argument cannot mix layer indices and regex strings.
        with pytest.raises(ValueError):
            _ = cached_transformers.get("bert-base-cased", True, reinit_modules=(1, "attentions"))
        # This model has a non-standard structure, so if a layer index or list of layer indexes
        # is provided, we raise a ConfigurationError.
        with pytest.raises(ConfigurationError):
            _ = cached_transformers.get("sshleifer/tiny-gpt2", True, reinit_modules=1)
        with pytest.raises(ConfigurationError):
            _ = cached_transformers.get("sshleifer/tiny-gpt2", True, reinit_modules=(1, 2))

    def test_reinit_modules_with_regex_strings(self):
        # Comparing all weights of the model is rather complicated, so arbitrarily compare the
        # weights of wpe module.
        reinit_module = "wpe"
        # This MUST be a deep copy, otherwise the parameters will be re-initialized and the
        # test will break.
        preinit_weights = list(
            cached_transformers.get("sshleifer/tiny-gpt2", True)
            .get_submodule(reinit_module)
            .parameters()
        )

        postinit_weights = list(
            cached_transformers.get(
                "sshleifer/tiny-gpt2",
                True,
                reinit_modules=(reinit_module,),
            )
            .get_submodule(reinit_module)
            .parameters()
        )
        assert all(
            (not torch.equal(pre, post) for pre, post in zip(preinit_weights, postinit_weights))
        )

    def test_from_pretrained_no_load_weights(self):
        _ = cached_transformers.get(
            "epwalsh/bert-xsmall-dummy", False, load_weights=False, cache_dir=self.TEST_DIR
        )
        # check that only three files were downloaded (filename.json, filename, filename.lock), for config.json
        # if more than three files were downloaded, then model weights were also (incorrectly) downloaded
        # NOTE: downloaded files are not explicitly detailed in Huggingface's public API,
        # so this assertion could fail in the future
        json_fnames = [fname for fname in os.listdir(str(self.TEST_DIR)) if fname.endswith(".json")]
        assert len(json_fnames) == 1
        json_data = json.load(open(str(self.TEST_DIR / json_fnames[0])))
        assert (
            json_data["url"]
            == "https://huggingface.co/epwalsh/bert-xsmall-dummy/resolve/main/config.json"
        )
        resource_id = os.path.splitext(json_fnames[0])[0]
        assert set(os.listdir(str(self.TEST_DIR))) == set(
            [json_fnames[0], resource_id, resource_id + ".lock"]
        )

    def test_from_pretrained_no_load_weights_local_config(self):
        config = AutoConfig.from_pretrained("epwalsh/bert-xsmall-dummy", cache_dir=self.TEST_DIR)
        self.clear_test_dir()

        # Save config to file.
        local_config_path = str(self.TEST_DIR / "local_config.json")
        config.to_json_file(local_config_path, use_diff=False)

        # Now load the model from the local config.
        _ = cached_transformers.get(
            local_config_path, False, load_weights=False, cache_dir=self.TEST_DIR
        )
        # Make sure no other files were downloaded.
        assert os.listdir(str(self.TEST_DIR)) == ["local_config.json"]

    def test_get_tokenizer_missing_from_cache_local_files_only(self):
        with pytest.raises((OSError, ValueError)):
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=self.TEST_DIR,
                local_files_only=True,
            )
