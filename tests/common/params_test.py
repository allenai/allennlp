import json
import os
import re
from collections import OrderedDict

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import (
    infer_and_cast,
    Params,
    remove_keys_from_params,
    with_overrides,
)
from allennlp.common.testing import AllenNlpTestCase


class TestParams(AllenNlpTestCase):
    def test_load_from_file(self):
        filename = self.FIXTURES_ROOT / "simple_tagger" / "experiment.json"
        params = Params.from_file(filename)

        assert "dataset_reader" in params
        assert "trainer" in params

        model_params = params.pop("model")
        assert model_params.pop("type") == "simple_tagger"

    def test_replace_none(self):
        params = Params({"a": "None", "b": [1.0, "None", 2], "c": {"d": "None"}})
        assert params["a"] is None
        assert params["b"][1] is None
        assert params["c"]["d"] is None

    def test_bad_unicode_environment_variables(self):
        filename = self.FIXTURES_ROOT / "simple_tagger" / "experiment.json"
        os.environ["BAD_ENVIRONMENT_VARIABLE"] = "\udce2"
        Params.from_file(filename)
        del os.environ["BAD_ENVIRONMENT_VARIABLE"]

    def test_with_overrides(self):
        original = {
            "foo": {"bar": {"baz": 3}, "x": 0},
            "bar": ["a", "b", "c"],
            "baz": {"bar": 2, "y": 3, "x": [0, 1, 2]},
        }
        overrides = {
            "foo.bar": {"z": 2},
            "bar.0": "d",
            "baz.bar": 1,
            "baz.x": [0, 0],
            "z": 2,
        }
        assert with_overrides(original, overrides) == {
            "foo": {"bar": {"z": 2}, "x": 0},
            "bar": ["d", "b", "c"],
            "baz": {"bar": 1, "y": 3, "x": [0, 0]},
            "z": 2,
        }

    def test_bad_overrides(self):
        with pytest.raises(ValueError, match="contains unused keys"):
            with_overrides({"foo": [0, 1, 2]}, {"foo.3": 4})
        with pytest.raises(ValueError, match="expected list or dict"):
            with_overrides({"foo": 3}, {"foo.x": 2})

    @pytest.mark.parametrize("input_type", [dict, str])
    def test_overrides(self, input_type):
        filename = self.FIXTURES_ROOT / "simple_tagger" / "experiment.json"
        overrides = {
            "train_data_path": "FOO",
            "model.type": "BAR",
            "model.text_field_embedder.token_embedders.tokens.type": "BAZ",
            "data_loader.batch_sampler.sorting_keys.0": "question",
        }
        params = Params.from_file(
            filename, overrides if input_type == dict else json.dumps(overrides)
        )

        assert "dataset_reader" in params
        assert "trainer" in params
        assert params["train_data_path"] == "FOO"
        assert params["data_loader"]["batch_sampler"]["sorting_keys"][0] == "question"

        model_params = params.pop("model")
        assert model_params.pop("type") == "BAR"
        assert model_params["text_field_embedder"]["token_embedders"]["tokens"]["type"] == "BAZ"

    def test_as_flat_dict(self):
        params = Params({"a": 10, "b": {"c": 20, "d": "stuff"}}).as_flat_dict()

        assert params == {"a": 10, "b.c": 20, "b.d": "stuff"}

    def test_jsonnet_features(self):
        config_file = self.TEST_DIR / "config.jsonnet"
        with open(config_file, "w") as f:
            f.write(
                """{
                            // This example is copied straight from the jsonnet docs
                            person1: {
                                name: "Alice",
                                welcome: "Hello " + self.name + "!",
                            },
                            person2: self.person1 { name: "Bob" },
                        }"""
            )

        params = Params.from_file(config_file)

        alice = params.pop("person1")
        bob = params.pop("person2")

        assert alice.as_dict() == {"name": "Alice", "welcome": "Hello Alice!"}
        assert bob.as_dict() == {"name": "Bob", "welcome": "Hello Bob!"}

        params.assert_empty("TestParams")

    def test_regexes_with_backslashes(self):
        bad_regex = self.TEST_DIR / "bad_regex.jsonnet"
        good_regex = self.TEST_DIR / "good_regex.jsonnet"

        with open(bad_regex, "w") as f:
            f.write(r'{"myRegex": "a\.b"}')

        with open(good_regex, "w") as f:
            f.write(r'{"myRegex": "a\\.b"}')

        with pytest.raises(RuntimeError):
            Params.from_file(bad_regex)

        params = Params.from_file(good_regex)
        regex = params["myRegex"]

        assert re.match(regex, "a.b")
        assert not re.match(regex, "a-b")

        # Check roundtripping
        good_regex2 = self.TEST_DIR / "good_regex2.jsonnet"
        with open(good_regex2, "w") as f:
            f.write(json.dumps(params.as_dict()))
        params2 = Params.from_file(good_regex2)

        assert params.as_dict() == params2.as_dict()

    def test_env_var_substitution(self):
        substitutor = self.TEST_DIR / "substitutor.jsonnet"
        key = "TEST_ENV_VAR_SUBSTITUTION"

        assert os.environ.get(key) is None

        with open(substitutor, "w") as f:
            f.write(f'{{"path": std.extVar("{key}")}}')

        # raises without environment variable set
        with pytest.raises(RuntimeError):
            Params.from_file(substitutor)

        os.environ[key] = "PERFECT"

        params = Params.from_file(substitutor)
        assert params["path"] == "PERFECT"

        del os.environ[key]

    @pytest.mark.xfail(
        not os.path.exists(AllenNlpTestCase.PROJECT_ROOT / "training_config"),
        reason="Training configs not installed with pip",
    )
    def test_known_configs(self):
        configs = os.listdir(self.PROJECT_ROOT / "training_config")

        # Our configs use environment variable substitution, and the _jsonnet parser
        # will fail if we don't pass it correct environment variables.
        forced_variables = [
            # constituency parser
            "PTB_TRAIN_PATH",
            "PTB_DEV_PATH",
            "PTB_TEST_PATH",
            # dependency parser
            "PTB_DEPENDENCIES_TRAIN",
            "PTB_DEPENDENCIES_VAL",
            # multilingual dependency parser
            "TRAIN_PATHNAME",
            "DEV_PATHNAME",
            "TEST_PATHNAME",
            # srl_elmo_5.5B
            "SRL_TRAIN_DATA_PATH",
            "SRL_VALIDATION_DATA_PATH",
            # coref
            "COREF_TRAIN_DATA_PATH",
            "COREF_DEV_DATA_PATH",
            "COREF_TEST_DATA_PATH",
            # ner
            "NER_TRAIN_DATA_PATH",
            "NER_TEST_A_PATH",
            "NER_TEST_B_PATH",
            # bidirectional lm
            "BIDIRECTIONAL_LM_TRAIN_PATH",
            "BIDIRECTIONAL_LM_VOCAB_PATH",
            "BIDIRECTIONAL_LM_ARCHIVE_PATH",
        ]

        for var in forced_variables:
            os.environ[var] = os.environ.get(var) or str(self.TEST_DIR)

        for config in configs:
            try:
                Params.from_file(self.PROJECT_ROOT / "training_config" / config)
            except Exception as e:
                raise AssertionError(f"unable to load params for {config}, because {e}")

        for var in forced_variables:
            if os.environ[var] == str(self.TEST_DIR):
                del os.environ[var]

    def test_as_ordered_dict(self):
        # keyD > keyC > keyE; keyDA > keyDB; Next all other keys alphabetically
        preference_orders = [["keyD", "keyC", "keyE"], ["keyDA", "keyDB"]]
        params = Params(
            {
                "keyC": "valC",
                "keyB": "valB",
                "keyA": "valA",
                "keyE": "valE",
                "keyD": {"keyDB": "valDB", "keyDA": "valDA"},
            }
        )
        ordered_params_dict = params.as_ordered_dict(preference_orders)
        expected_ordered_params_dict = OrderedDict(
            {
                "keyD": {"keyDA": "valDA", "keyDB": "valDB"},
                "keyC": "valC",
                "keyE": "valE",
                "keyA": "valA",
                "keyB": "valB",
            }
        )
        assert json.dumps(ordered_params_dict) == json.dumps(expected_ordered_params_dict)

    def test_to_file(self):
        # Test to_file works with or without preference orders
        params_dict = {"keyA": "valA", "keyB": "valB"}
        expected_ordered_params_dict = OrderedDict({"keyB": "valB", "keyA": "valA"})
        params = Params(params_dict)
        file_path = self.TEST_DIR / "config.jsonnet"
        # check with preference orders
        params.to_file(file_path, [["keyB", "keyA"]])
        with open(file_path, "r") as handle:
            ordered_params_dict = OrderedDict(json.load(handle))
        assert json.dumps(expected_ordered_params_dict) == json.dumps(ordered_params_dict)
        # check without preference orders doesn't give error
        params.to_file(file_path)

    def test_infer_and_cast(self):
        lots_of_strings = {
            "a": ["10", "1.3", "true"],
            "b": {"x": 10, "y": "20.1", "z": "other things"},
            "c": "just a string",
        }

        casted = {
            "a": [10, 1.3, True],
            "b": {"x": 10, "y": 20.1, "z": "other things"},
            "c": "just a string",
        }

        assert infer_and_cast(lots_of_strings) == casted

        contains_bad_data = {"x": 10, "y": int}
        with pytest.raises(ValueError, match="cannot infer type"):
            infer_and_cast(contains_bad_data)

        params = Params(lots_of_strings)

        assert params.as_dict() == lots_of_strings
        assert params.as_dict(infer_type_and_cast=True) == casted

    def test_pop_choice(self):
        choices = ["my_model", "other_model"]
        params = Params({"model": "my_model"})
        assert params.pop_choice("model", choices) == "my_model"

        params = Params({"model": "non_existent_model"})
        with pytest.raises(ConfigurationError):
            params.pop_choice("model", choices)

        params = Params({"model": "module.submodule.ModelName"})
        assert params.pop_choice("model", "choices") == "module.submodule.ModelName"

        params = Params({"model": "module.submodule.ModelName"})
        with pytest.raises(ConfigurationError):
            params.pop_choice("model", choices, allow_class_names=False)

    def test_remove_keys_from_params(self):
        filename = self.FIXTURES_ROOT / "simple_tagger" / "experiment.json"
        params = Params.from_file(filename)

        assert params["data_loader"]["batch_sampler"]["type"] == "bucket"
        assert params["data_loader"]["batch_sampler"]["batch_size"] == 80

        remove_keys_from_params(params, keys=["batch_size"])
        assert "batch_size" not in params["data_loader"]["batch_sampler"]

        remove_keys_from_params(params, keys=["type", "batch_size"])
        assert "type" not in params["data_loader"]["batch_sampler"]

        remove_keys_from_params(params, keys=["data_loader"])
        assert "data_loader" not in params
