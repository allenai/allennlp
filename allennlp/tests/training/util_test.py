# pylint: disable=invalid-name,too-many-public-methods,protected-access
import json
import os
import shutil

from allennlp.common import Params
from allennlp.common.util import flatten_filename
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training import util


class TestTrainerUtil(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.snli_file = str(self.FIXTURES_ROOT / "data" / "snli.jsonl")
        self.params = Params({"dataset_reader": {"type": "snli"}, "train_data_path": self.snli_file})
        self.cache_directory = str(self.FIXTURES_ROOT / "data_cache")

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.cache_directory)

    def test_datasets_from_params_uses_caching_correctly_in_simplest_case(self):
        # We'll rely on the dataset reader tests to be sure of the functionality of this caching;
        # we're just checking here that things get hooked up correctly to the right spots.
        cache_prefix = "prefix"
        _ = util.datasets_from_params(self.params.duplicate(), self.cache_directory, cache_prefix)

        expected_cache_file = f"{self.cache_directory}/{cache_prefix}/{flatten_filename(self.snli_file)}"
        expected_param_file = f"{self.cache_directory}/{cache_prefix}/params.json"
        assert os.path.exists(expected_cache_file)
        assert os.path.exists(expected_param_file)
        with open(expected_param_file, 'r') as param_file:
            saved_params = json.load(param_file)
            assert saved_params == self.params.pop('dataset_reader').as_dict(quiet=True)

    def test_datasets_from_params_uses_caching_correctly_with_hashed_params(self):
        # We'll rely on the dataset reader tests to be sure of the functionality of this caching;
        # we're just checking here that things get hooked up correctly to the right spots.
        _ = util.datasets_from_params(self.params, self.cache_directory)

        cache_prefix = util._dataset_reader_param_hash(Params({"type": "snli"}))
        expected_cache_file = f"{self.cache_directory}/{cache_prefix}/{flatten_filename(self.snli_file)}"
        assert os.path.exists(expected_cache_file)
