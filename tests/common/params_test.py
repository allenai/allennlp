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
