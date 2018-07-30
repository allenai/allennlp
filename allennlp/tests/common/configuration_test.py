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

        # Is OK with correct configurations.
        params = new_params()
        assert not find_errors(params)

        # Complains about missing requireds
        params = new_params()
        params.pop('model')
        errors = find_errors(params)
        assert errors == ['key model is required but was not specified']

        # Complains about mispecified ints
        params = new_params()
        params['trainer']['num_epochs'] = "oops"
        errors = find_errors(params)
        assert errors == ['expected int value at key trainer.num_epochs, got oops']

        # Complains about mispecified floats
        params = new_params()
        params['trainer']['grad_norm'] = "norm"
        errors = find_errors(params)
        assert errors == ['expected float value at key trainer.grad_norm, got norm']

        # Complains about mispecified bools
        params = new_params()
        params['iterator']['track_epoch'] = 1
        errors = find_errors(params)
        assert errors == ['expected bool value at key iterator.track_epoch, got 1']

        # Complains about extra parameters
        params = new_params()
        params['trainer']['unnecessary'] = 0
        errors = find_errors(params)
        assert errors == ["extra keys provided for trainer: {'unnecessary'}"]
