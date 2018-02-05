# pylint: disable=no-self-use,invalid-name
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase


class TestParams(AllenNlpTestCase):

    def test_load_from_file(self):
        filename = 'tests/fixtures/bidaf/experiment.json'
        params = Params.from_file(filename)

        assert "dataset_reader" in params
        assert "trainer" in params

        model_params = params.pop("model")
        assert model_params.pop("type") == "bidaf"

    def test_overrides(self):
        filename = 'tests/fixtures/bidaf/experiment.json'
        overrides = '{ "train_data_path": "FOO", "model": { "type": "BAR" },'\
                    'model.text_field_embedder.tokens.type: "BAZ" }'
        params = Params.from_file(filename, overrides)

        assert "dataset_reader" in params
        assert "trainer" in params
        assert params["train_data_path"] == "FOO"

        model_params = params.pop("model")
        assert model_params.pop("type") == "BAR"
        assert model_params["text_field_embedder.tokens.type"] == "BAZ"

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
                                "filename": "my-file",
                                "c": {
                                        "c_file": "my-other-file"
                                }
                        }
                }
        })

        # Construct ``A`` from params but then just throw it away.
        A.from_params(params.pop("a"))

        assert params.files_to_archive == {
                "a.b.filename": "my-file",
                "a.b.c.c_file": "my-other-file"
        }
