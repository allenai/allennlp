# pylint: disable=no-self-use,invalid-name
import os

import pytest

from allennlp.common.configuration import configure, Config, BASE_CONFIG, find_errors
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase



class TestConfiguration(AllenNlpTestCase):
    def test_configure_top_level(self):
        config = configure()

        assert config == BASE_CONFIG

    def test_abstract_base_class(self):
        config = configure('allennlp.data.dataset_readers.dataset_reader.DatasetReader')

        assert isinstance(config, list)
        assert 'allennlp.data.dataset_readers.snli.SnliReader' in config

    def test_specific_subclass(self):
        config = configure('allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        assert isinstance(config, Config)

        items = {item.name: item for item in config.items}

        assert len(items) == 3

        assert 'token_indexers' in items
        token_indexers = items['token_indexers']
        assert token_indexers.default_value is None

        assert 'domain_identifier' in items
        domain_identifier = items['domain_identifier']
        assert domain_identifier.annotation == str
        assert domain_identifier.default_value is None

        assert 'lazy' in items
        lazy = items['lazy']
        assert lazy.annotation == bool
        assert not lazy.default_value

    def test_errors(self):
        with pytest.raises(ModuleNotFoundError):
            configure('allennlp.non_existent_module.SomeClass')

        with pytest.raises(AttributeError):
            configure('allennlp.data.dataset_readers.NonExistentDatasetReader')

    def test_validation(self):
        new_params = lambda: Params.from_file(self.PROJECT_ROOT / 'allennlp' / 'tests' / 'fixtures' / 'simple_tagger' / 'experiment.json')

        # Works as is
        params = new_params()
        assert not find_errors(params)

        # Complains about missing requireds
        params = new_params()
        params.pop('model')
        errors = find_errors(params)
        assert errors == ['key model is required but was not specified']

        # Complains about wrong types
        params = new_params()
        params['trainer']['num_epochs'] = "oops"
        errors = find_errors(params)
        assert errors
