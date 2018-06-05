# pylint: disable=no-self-use,invalid-name,line-too-long
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

    def test_root_config(self):
        response = self.client.get('/')
        html = response.get_data().decode('utf-8')

        assert "allennlp.data.vocabulary.Vocabulary" in html
        assert "/?class=allennlp.data.vocabulary.Vocabulary" in html

    def test_choices(self):
        response = self.client.get('/?class=allennlp.data.dataset_readers.dataset_reader.DatasetReader')
        html = response.get_data().decode('utf-8')

        assert "allennlp.data.dataset_readers.semantic_role_labeling.SrlReader" in html
        assert "/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader" in html

    def test_subclass(self):
        response = self.client.get('/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        html = response.get_data().decode('utf-8')

        assert '"type": "srl"' in html
        assert '// "token_indexers"' in html

    def test_torch_class(self):
        response = self.client.get('/?class=torch.optim.rmsprop.RMSprop')
        html = response.get_data().decode('utf-8')

        assert '"type": "rmsprop"' in html
        assert '// "weight_decay"' in html

    def test_rnn_hack(self):
        response = self.client.get('/?class=torch.nn.modules.rnn.LSTM')
        html = response.get_data().decode('utf-8')

        assert '"type": "lstm"' in html
        assert '// "batch_first"' in html

    def test_initializers(self):
        response = self.client.get('/?class=allennlp.nn.initializers.Initializer')
        html = response.get_data().decode('utf-8')

        assert 'torch.nn.init.constant_' in html
        assert '/?class=torch.nn.init.constant_' in html
        assert 'allennlp.nn.initializers.block_orthogonal' in html
        assert '/?class=allennlp.nn.initializers.block_orthogonal' in html

        response = self.client.get('/?class=torch.nn.init.uniform_')
        html = response.get_data().decode('utf-8')
        assert '"type": "uniform"' in html
        assert '// "a":' in html

    def test_regularizers(self):
        response = self.client.get('/?class=allennlp.nn.regularizers.regularizer.Regularizer')
        html = response.get_data().decode('utf-8')

        assert 'allennlp.nn.regularizers.regularizers.L1Regularizer' in html
        assert '/?class=allennlp.nn.regularizers.regularizers.L1Regularizer' in html

        response = self.client.get('/?class=allennlp.nn.regularizers.regularizers.L1Regularizer')
        html = response.get_data().decode('utf-8')
        assert '"type": "l1"' in html
        assert '// "alpha":' in html

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
        response = client.get('/?class=allennlp.predictors.predictor.Predictor')
        html = response.get_data().decode('utf-8')
        assert "allennlp.predictors.bidaf.BidafPredictor" in html
        assert "configexplorer.predictor.BidafPredictor" not in html

        # With specifying extra modules, it should be there.
        app = make_app(['configexplorer'])
        app.testing = True
        client = app.test_client()
        response = client.get('/?class=allennlp.predictors.predictor.Predictor')
        html = response.get_data().decode('utf-8')
        assert "allennlp.predictors.bidaf.BidafPredictor" in html
        assert "configexplorer.predictor.BidafPredictor" in html

        sys.path.remove(str(self.TEST_DIR))
