# pylint: disable=no-self-use,invalid-name,line-too-long

import json
import os
import sys

from allennlp.common.testing import AllenNlpTestCase
from allennlp.service.config_explorer import make_app


class TestConfigExplorer(AllenNlpTestCase):

    def setUp(self):
        super().setUp()
        app = make_app()
        app.testing = True
        self.client = app.test_client()

    def test_app(self):
        response = self.client.get('/')
        html = response.get_data().decode('utf-8')

        assert "AllenNLP Configuration Wizard" in html

    def test_api(self):
        response = self.client.get('/api/')
        data = json.loads(response.get_data())

        assert data["className"] == ""

        items = data["configItems"]
        assert items

        assert items[0] == {
                "name": "dataset_reader",
                "configurable": True,
                "comment": "specify your dataset reader here",
                "annotation": ["allennlp.data.dataset_readers.dataset_reader.DatasetReader"]
        }


    def test_choices(self):
        response = self.client.get('/api/?class=allennlp.data.dataset_readers.dataset_reader.DatasetReader')
        data = json.loads(response.get_data())

        assert "allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader" in data["choices"]

    def test_subclass(self):
        response = self.client.get('/api/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        data = json.loads(response.get_data())

        items = data['configItems']
        assert items[0] == {"name": "type", "type": "srl"}
        assert items[1]["name"] == "token_indexers"

    def test_torch_class(self):
        response = self.client.get('/api/?class=torch.optim.rmsprop.RMSprop')
        data = json.loads(response.get_data())
        items = data['configItems']

        assert items[0]["type"] == "rmsprop"
        assert any(item["name"] == "lr" for item in items)

    def test_rnn_hack(self):
        response = self.client.get('/api/?class=torch.nn.modules.rnn.LSTM')
        data = json.loads(response.get_data())
        items = data['configItems']

        assert items[0]["type"] == "lstm"
        assert any(item["name"] == "batch_first" for item in items)

    def test_initializers(self):
        response = self.client.get('/api/?class=allennlp.nn.initializers.Initializer')
        data = json.loads(response.get_data())

        assert 'torch.nn.init.constant_' in data["choices"]
        assert 'allennlp.nn.initializers.block_orthogonal' in data["choices"]

        response = self.client.get('/api/?class=torch.nn.init.uniform_')
        data = json.loads(response.get_data())
        items = data['configItems']

        assert items[0]["type"] == "uniform"
        assert any(item["name"] == "a" for item in items)

    def test_regularizers(self):
        response = self.client.get('/api/?class=allennlp.nn.regularizers.regularizer.Regularizer')
        data = json.loads(response.get_data())

        assert 'allennlp.nn.regularizers.regularizers.L1Regularizer' in data["choices"]

        response = self.client.get('/api/?class=allennlp.nn.regularizers.regularizers.L1Regularizer')
        data = json.loads(response.get_data())
        items = data['configItems']

        assert items[0]["type"] == "l1"
        assert any(item["name"] == "alpha" for item in items)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / 'configexplorer'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / '__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, str(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace("""@Predictor.register('machine-comprehension')""",
                                    """@Predictor.register('config-explorer-predictor')""")

        with open(os.path.join(packagedir, 'predictor.py'), 'w') as f:
            f.write(code)

        # Without specifying modules to load, it shouldn't be there
        app = make_app()
        app.testing = True
        client = app.test_client()
        response = client.get('/api/?class=allennlp.predictors.predictor.Predictor')
        data = json.loads(response.get_data())
        assert "allennlp.predictors.bidaf.BidafPredictor" in data["choices"]
        assert "configexplorer.predictor.BidafPredictor" not in data["choices"]

        # With specifying extra modules, it should be there.
        app = make_app(['configexplorer'])
        app.testing = True
        client = app.test_client()
        response = client.get('/api/?class=allennlp.predictors.predictor.Predictor')
        data = json.loads(response.get_data())
        assert "allennlp.predictors.bidaf.BidafPredictor" in data["choices"]
        assert "configexplorer.predictor.BidafPredictor" in data["choices"]

        sys.path.remove(str(self.TEST_DIR))
