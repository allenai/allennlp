# pylint: disable=invalid-name,protected-access
import logging
import os
import sys
import shutil
from unittest import TestCase

import torch
import pytest
from numpy.testing import assert_allclose

from allennlp.commands.train import train_model_from_file
from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params
from allennlp.data import Dataset, DatasetReader
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models import Model, load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True
        logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
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

    def ensure_model_can_train_save_and_load(self, param_file: str):
        save_dir = os.path.join(self.TEST_DIR, "save_and_load_test")
        archive_file = os.path.join(save_dir, "model.tar.gz")
        model = train_model_from_file(param_file, save_dir)
        loaded_model = load_archive(archive_file).model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # First we make sure that the state dict (the parameters) are the same for both models.
        for key in state_keys:
            assert_allclose(model.state_dict()[key].numpy(),
                            loaded_model.state_dict()[key].numpy(),
                            err_msg=key)
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])
        iterator = DataIterator.from_params(params['iterator'])

        # We'll check that even if we index the dataset with each model separately, we still get
        # the same result out.
        model_dataset = reader.read(params['validation_data_path'])
        model_dataset.index_instances(model.vocab)
        model_batch_arrays = next(iterator(model_dataset, shuffle=False))
        model_batch = arrays_to_variables(model_batch_arrays, for_training=False)
        loaded_dataset = reader.read(params['validation_data_path'])
        loaded_dataset.index_instances(loaded_model.vocab)
        loaded_batch_arrays = next(iterator(loaded_dataset, shuffle=False))
        loaded_batch = arrays_to_variables(loaded_batch_arrays, for_training=False)

        # The datasets themselves should be identical.
        for key in model_batch.keys():
            field = model_batch[key]
            if isinstance(field, dict):
                for subfield in field:
                    assert_allclose(model_batch[key][subfield].data.numpy(),
                                    loaded_batch[key][subfield].data.numpy(),
                                    rtol=1e-6,
                                    err_msg=key + "." + subfield)
            else:
                assert_allclose(model_batch[key].data.numpy(),
                                loaded_batch[key].data.numpy(),
                                rtol=1e-6,
                                err_msg=key)

        # Set eval mode, to turn off things like dropout, then get predictions.
        model.eval()
        loaded_model.eval()
        model_predictions = model.forward(**model_batch)
        loaded_model_predictions = loaded_model.forward(**loaded_batch)

        # Check loaded model's loss exists and we can compute gradients, for continuing training.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values for these keys should be close.
        for key in model_predictions.keys():
            assert_allclose(model_predictions[key].data.numpy(),
                            loaded_model_predictions[key].data.numpy(),
                            rtol=1e-6,
                            err_msg=key)

        return model, loaded_model
