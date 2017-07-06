# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp import NlpApi


class TestNlpApi(AllenNlpTestCase):
    # Most of these tests use `_get_module_from_dict` directly, because that's where the bulk of
    # the implementation code is.  We also have a few tests that make sure the constructor
    # parameters get hooked up correctly to the `_get_module_from_dict` function with the various
    # API calls.
    # pylint: disable=protected-access
    def test_get_module_from_dict_returns_object_if_name_is_present(self):
        module = NlpApi._get_module_from_dict({'name': 1}, None, 'name', "doesn't matter", None)
        assert module == 1

    def test_get_module_from_dict_crashes_when_name_not_present(self):
        with pytest.raises(ConfigurationError, message="No test module specified for name: name"):
            _ = NlpApi._get_module_from_dict({}, None, 'name', "crash", "test")

    def test_get_module_from_dict_crashes_with_bad_fallback_behavior(self):
        with pytest.raises(ConfigurationError, message="Unrecognized fallback behavior: fake"):
            _ = NlpApi._get_module_from_dict({}, None, 'name', "fake", "test")

    def test_get_module_from_dict_uses_default_fn_correctly(self):
        i = 0
        def default_fn():
            nonlocal i
            i += 1
            return i
        module_dict = {}
        module1a = NlpApi._get_module_from_dict(module_dict, default_fn, 'name1', "new default", None)
        module1b = NlpApi._get_module_from_dict(module_dict, default_fn, 'name1', "new default", None)
        module2 = NlpApi._get_module_from_dict(module_dict, default_fn, 'name2', "new default", None)
        module3 = NlpApi._get_module_from_dict(module_dict, default_fn, 'name3', "new default", None)
        assert module1a == 1
        assert module1b == 1
        assert module2 == 2
        assert module3 == 3

    def test_get_module_from_dict_reuses_layers_correctly(self):
        i = 0
        def default_fn():
            nonlocal i
            i += 1
            return i
        module_dict = {'name2': object()}
        module1 = NlpApi._get_module_from_dict(module_dict, default_fn, 'name1', "new default", None)
        module2 = NlpApi._get_module_from_dict(module_dict, default_fn, 'name2', "use name1", None)
        module3a = NlpApi._get_module_from_dict(module_dict, default_fn, 'name3', "use name1", None)
        module3b = NlpApi._get_module_from_dict(module_dict, default_fn, 'name3', "new default", None)
        assert module1 == 1
        assert module2 is module_dict['name2']
        assert module3a == 1
        assert module3b == 1

    def test_get_token_embedder_uses_constructor_arguments_correctly(self):
        api = NlpApi(token_embedders={'default': 1}, default_token_embedder=lambda: 2)
        assert api.get_token_embedder() == 1
        assert api.get_token_embedder('name', fallback_behavior='new default') == 2
        with pytest.raises(ConfigurationError, message='No token embedder module specified for name: x'):
            api.get_token_embedder('x')

    def test_get_context_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(context_encoders={'default': 1}, default_context_encoder=lambda: 2)
        assert api.get_context_encoder('default') == 1
        assert api.get_context_encoder('name', fallback_behavior='new default') == 2
        with pytest.raises(ConfigurationError, message='No context encoder module specified for name: x'):
            api.get_context_encoder('x')

    def test_get_sentence_encoder_uses_constructor_arguments_correctly(self):
        api = NlpApi(sentence_encoders={'default': 1}, default_sentence_encoder=lambda: 2)
        assert api.get_sentence_encoder('default') == 1
        assert api.get_sentence_encoder('name', fallback_behavior='new default') == 2
        with pytest.raises(ConfigurationError, message='No sentence encoder module specified for name: x'):
            api.get_sentence_encoder('x')
