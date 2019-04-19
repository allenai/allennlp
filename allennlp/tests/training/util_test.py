# pylint: disable=invalid-name,too-many-public-methods,protected-access
import os
import shutil

from allennlp.common import Params
from allennlp.common import util as common_util
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training import util


class TestTrainerUtil(AllenNlpTestCase):
    def test_datasets_from_params_uses_caching_correctly_in_simplest_case(self):
        # We'll rely on the dataset reader tests to be sure of the functionality of this caching;
        # we're just checking here that things get hooked up correctly to the right spots.
        snli_file = str(self.FIXTURES_ROOT / "data" / "snli.jsonl")
        params = Params({"dataset_reader": {"type": "snli"}, "train_data_path": snli_file})
        cache_directory = str(self.FIXTURES_ROOT / "data_cache")
        cache_prefix = "prefix"
        _ = util.datasets_from_params(params, cache_directory, cache_prefix)

        expected_cache_file = f"{cache_directory}/{cache_prefix}/{common_util.flatten_filename(snli_file)}"
        try:
            assert os.path.exists(expected_cache_file)
        finally:
            shutil.rmtree(cache_directory)

    def test_datasets_from_params_uses_caching_correctly_with_hashed_params(self):
        # We'll rely on the dataset reader tests to be sure of the functionality of this caching;
        # we're just checking here that things get hooked up correctly to the right spots.
        snli_file = str(self.FIXTURES_ROOT / "data" / "snli.jsonl")
        params = Params({"dataset_reader": {"type": "snli"}, "train_data_path": snli_file})
        cache_directory = str(self.FIXTURES_ROOT / "data_cache")
        _ = util.datasets_from_params(params, cache_directory)

        cache_prefix = util._dataset_reader_param_hash(Params({"type": "snli"}))
        expected_cache_file = f"{cache_directory}/{cache_prefix}/{common_util.flatten_filename(snli_file)}"
        try:
            assert os.path.exists(expected_cache_file)
        finally:
            shutil.rmtree(cache_directory)
