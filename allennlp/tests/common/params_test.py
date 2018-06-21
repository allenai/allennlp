# pylint: disable=no-self-use,invalid-name,bad-continuation
import os
import tempfile

from allennlp.common.params import Params, unflatten, with_fallback, parse_overrides
from allennlp.common.testing import AllenNlpTestCase


class TestParams(AllenNlpTestCase):

    def test_load_from_file(self):
        filename = self.FIXTURES_ROOT / 'bidaf' / 'experiment.json'
        params = Params.from_file(filename)

        assert "dataset_reader" in params
        assert "trainer" in params

        model_params = params.pop("model")
        assert model_params.pop("type") == "bidaf"

    def test_overrides(self):
        filename = self.FIXTURES_ROOT / 'bidaf' / 'experiment.json'
        overrides = '{ "train_data_path": "FOO", "model": { "type": "BAR" },'\
                    '"model.text_field_embedder.tokens.type": "BAZ" }'
        params = Params.from_file(filename, overrides)

        assert "dataset_reader" in params
        assert "trainer" in params
        assert params["train_data_path"] == "FOO"

        model_params = params.pop("model")
        assert model_params.pop("type") == "BAR"
        assert model_params["text_field_embedder"]["tokens"]["type"] == "BAZ"

    def test_unflatten(self):
        flattened = {"a.b.c": 1, "a.b.d": 0, "a.e.f.g.h": 2, "b": 3}
        unflattened = unflatten(flattened)
        assert unflattened == {
            "a": {
                "b": {
                    "c": 1,
                    "d": 0
                },
                "e": {
                    "f": {
                        "g": {
                            "h": 2
                        }
                    }
                }
            },
            "b": 3
        }

        # should do nothing to a non-flat dictionary
        assert unflatten(unflattened) == unflattened

    def test_with_fallback(self):
        preferred = {"a": 1}
        fallback = {"a": 0, "b": 2}

        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {"a": 1, "b": 2}

        # incompatibility is ok
        preferred = {"a": {"c": 3}}
        fallback = {"a": 0, "b": 2}
        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {"a": {"c": 3}, "b": 2}

        # goes deep
        preferred = {"deep": {"a": 1}}
        fallback = {"deep": {"a": 0, "b": 2}}

        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {"deep": {"a": 1, "b": 2}}

    def test_parse_overrides(self):
        assert parse_overrides("") == {}
        assert parse_overrides("{}") == {}

        override_dict = parse_overrides('{"train_data": "/train", "trainer.num_epochs": 10}')
        assert override_dict == {
            "train_data": "/train",
            "trainer": {
                "num_epochs": 10
            }
        }

        params = with_fallback(
            preferred=override_dict,
            fallback={
                "train_data": "/test",
                "model": "bidaf",
                "trainer": {"num_epochs": 100, "optimizer": "sgd"}
            })

        assert params == {
            "train_data": "/train",
            "model": "bidaf",
            "trainer": {"num_epochs": 10, "optimizer": "sgd"}
        }

    def test_as_flat_dict(self):
        params = Params({
                'a': 10,
                'b': {
                        'c': 20,
                        'd': 'stuff'
                }
        }).as_flat_dict()

        assert params == {'a': 10, 'b.c': 20, 'b.d': 'stuff'}

    def test_add_file_to_archive(self):
        # Creates actual files since add_file_to_archive will throw an exception
        # if the file does not exist.
        tempdir = tempfile.mkdtemp()
        my_file = os.path.join(tempdir, "my_file.txt")
        my_other_file = os.path.join(tempdir, "my_other_file.txt")
        open(my_file, 'w').close()
        open(my_other_file, 'w').close()

        # Some nested classes just to exercise the ``from_params``
        # and ``add_file_to_archive`` methods.
        class A:
            def __init__(self, b: 'B') -> None:
                self.b = b

            @classmethod
            def from_params(cls, params: Params) -> 'A':
                b_params = params.pop("b")
                return cls(B.from_params(b_params))

        class B:
            def __init__(self, filename: str, c: 'C') -> None:
                self.filename = filename
                self.c_dict = {"here": c}

            @classmethod
            def from_params(cls, params: Params) -> 'B':
                params.add_file_to_archive("filename")

                filename = params.pop("filename")
                c_params = params.pop("c")
                c = C.from_params(c_params)

                return cls(filename, c)

        class C:
            def __init__(self, c_file: str) -> None:
                self.c_file = c_file

            @classmethod
            def from_params(cls, params: Params) -> 'C':
                params.add_file_to_archive("c_file")
                c_file = params.pop("c_file")

                return cls(c_file)


        params = Params({
                "a": {
                        "b": {
                                "filename": my_file,
                                "c": {
                                        "c_file": my_other_file
                                }
                        }
                }
        })

        # Construct ``A`` from params but then just throw it away.
        A.from_params(params.pop("a"))

        assert params.files_to_archive == {
                "a.b.filename": my_file,
                "a.b.c.c_file": my_other_file
        }
