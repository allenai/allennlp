# pylint: disable=invalid-name,protected-access
import logging
import os
import sys
import shutil
from unittest import TestCase

import torch
import pytest
from numpy.testing import assert_allclose

from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params, PARAMETER
from allennlp.data.dataset import Dataset
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'
    MODEL_FILE = TEST_DIR + "model.th"

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.INFO)
        logging.disable(PARAMETER)
        log_pytorch_version_info()
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    @staticmethod
    @pytest.fixture(scope='module')
    def unload_registrable():
        # Each test class which tests registrable subclasses registers the
        # subclasses when they are imported into the test file, which is
        # modifying a class attribute of "Registrable". In order to test that
        # subclasses are being correctly registered for each set of registrable
        # things, we want to clear the registry in the tests between each test.
        # However, if we do this between individual tests, we clear the registry
        # prematurely, as the imports at the top of a file containing a test class
        # are not re-run for every individual test within a test class, causing some
        # tests to fail as the registry does not have the correct keys.

        # This clears the registry in between each test file (due to the 'module'
        # level scope), preventing this behaviour and making the tests as
        # maximally isolated as feasible.

        del sys.modules['allennlp.common.registrable']

    def get_trainer_params(self, additional_arguments=None):
        params = Params({})
        params['save_models'] = False
        params['serialization_prefix'] = self.MODEL_FILE
        params['num_epochs'] = 1

        if additional_arguments:
            for key, value in additional_arguments.items():
                params[key] = value
        return params

    def ensure_model_can_train_save_and_load(self,
                                             model: Model,
                                             dataset: Dataset,
                                             iterator: DataIterator = None):
        data_iterator = iterator or BasicIterator()
        single_batch = next(data_iterator(dataset))
        single_batch = arrays_to_variables(single_batch)
        model_predictions = model.forward(**single_batch)

        # Check loss exists and we can compute gradients.
        model_loss = model_predictions["loss"]
        assert model_loss is not None
        model_loss.backward()

        torch.save(model.state_dict(), self.MODEL_FILE)
        loaded_model = model
        loaded_model.zero_grad()
        loaded_model.load_state_dict(torch.load(self.MODEL_FILE))
        loaded_model_predictions = loaded_model.forward(**single_batch)

        # Check loaded model's loss exists and we can compute gradients.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values
        # for these keys should be close.
        for key in model_predictions.keys():
            assert_allclose(model_predictions[key].data.numpy(), loaded_model_predictions[key].data.numpy())

        return model, loaded_model
