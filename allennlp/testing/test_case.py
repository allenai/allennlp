# pylint: disable=invalid-name,protected-access
from copy import deepcopy
from unittest import TestCase
import codecs
import gzip
import logging
import os
import shutil

from numpy.testing import assert_allclose

from ..common.checks import log_pytorch_version_info
from ..common.params import Params


class AllenNlpTestCase(TestCase):  # pylint: disable=too-many-public-methods
    TEST_DIR = './TMP_TEST/'
    TRAIN_FILE = TEST_DIR + 'train_file'
    VALIDATION_FILE = TEST_DIR + 'validation_file'
    TEST_FILE = TEST_DIR + 'test_file'
    TRAIN_BACKGROUND = TEST_DIR + 'train_background'
    VALIDATION_BACKGROUND = TEST_DIR + 'validation_background'
    SNLI_FILE = TEST_DIR + 'snli_file'
    PRETRAINED_VECTORS_FILE = TEST_DIR + 'pretrained_glove_vectors_file'
    PRETRAINED_VECTORS_GZIP = TEST_DIR + 'pretrained_glove_vectors_file.gz'

    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.DEBUG)
        log_pytorch_version_info()
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def get_model_params(self, additional_arguments=None):
        params = Params({})
        params['save_models'] = False
        params['model_serialization_prefix'] = self.TEST_DIR
        params['train_files'] = [self.TRAIN_FILE]
        params['validation_files'] = [self.VALIDATION_FILE]
        params['embeddings'] = {'words': {'dimension': 6}, 'characters': {'dimension': 2}}
        params['encoder'] = {"default": {'type': 'bow'}}
        params['num_epochs'] = 1
        params['validation_split'] = 0.0

        if additional_arguments:
            for key, value in additional_arguments.items():
                params[key] = deepcopy(value)
        return params

    def get_model(self, model_class, additional_arguments=None):
        params = self.get_model_params(additional_arguments)
        return model_class(params)

    def ensure_model_trains_and_loads(self, model_class, args: Params):
        args['save_models'] = True
        # Our loading tests work better if you're not using data generators.  Unless you
        # specifically request it in your test, we'll avoid using them here, and if you _do_ use
        # them, we'll skip some of the stuff below that isn't compatible.
        args.setdefault('data_generator', None)
        model = self.get_model(model_class, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(model_class, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        if model._uses_data_generators():
            # We shuffle the data in the data generator.  Instead of making that logic more
            # complicated, we'll just pass on the loading tests here.  See comment above.
            pass
        else:
            model_predictions = model.model.predict(model.validation_arrays[0])
            loaded_model_predictions = loaded_model.model.predict(model.validation_arrays[0])

            for model_prediction, loaded_prediction in zip(model_predictions, loaded_model_predictions):
                assert_allclose(model_prediction, loaded_prediction)

        # We should get the same result if we index the data from the original model and the loaded
        # model.
        _, indexed_validation_arrays = loaded_model.load_data_arrays(model.validation_files)
        if model._uses_data_generators():
            # As above, we'll just pass on this.
            pass
        else:
            model_predictions = model.model.predict(model.validation_arrays[0])
            loaded_model_predictions = loaded_model.model.predict(indexed_validation_arrays[0])

            for model_prediction, loaded_prediction in zip(model_predictions, loaded_model_predictions):
                assert_allclose(model_prediction, loaded_prediction)
        return model, loaded_model

    def write_pretrained_vector_files(self):
        # write the file
        with codecs.open(self.PRETRAINED_VECTORS_FILE, 'w', 'utf-8') as vector_file:
            vector_file.write('word2 0.21 0.57 0.51 0.31\n')
            vector_file.write('sentence1 0.81 0.48 0.19 0.47\n')
        # compress the file
        with open(self.PRETRAINED_VECTORS_FILE, 'rb') as f_in:
            with gzip.open(self.PRETRAINED_VECTORS_GZIP, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
