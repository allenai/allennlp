# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch

import pytest
import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import BidirectionalAttentionFlow, Model


class DialogQATest(ModelTestCase):
    def setUp(self):
        super(DialogQATest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'dialog_qa' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'quac_sample.json')
        self.batch = Batch(self.instances)
        self.batch.index_instances(self.vocab)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "best_span_str" in output_dict and "loss" in output_dict
        assert "followup" in output_dict and "yesno" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_batch_predictions_are_consistent(self):
        # Save some state.
        saved_model = self.model
        saved_instances = self.instances

        # Modify the state, run the test with modified state.
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])
        self.instances = reader.read(self.FIXTURES_ROOT / 'data' / 'quac_sample.json')
        vocab = Vocabulary.from_instances(self.instances)
        for instance in self.instances:
            instance.index_fields(vocab)
        self.model = Model.from_params(vocab=vocab, params=params['model'])
        self.ensure_batch_predictions_are_consistent()
