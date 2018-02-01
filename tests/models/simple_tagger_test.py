# pylint: disable=invalid-name,protected-access
from flaky import flaky
import pytest
import numpy

import torch

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator, BasicIterator
from allennlp.models import Model
from allennlp.training import Trainer

class SimpleTaggerTest(ModelTestCase):
    def setUp(self):
        super(SimpleTaggerTest, self).setUp()
        self.set_up_model('tests/fixtures/simple_tagger/experiment.json',
                          'tests/fixtures/data/sequence_tagging.tsv')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.decode(output_dict)
        class_probs = output_dict['class_probabilities'][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1), numpy.array([1, 1, 1, 1]))

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the stacked_encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["stacked_encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(self.vocab, params.pop("model"))

    def test_regularization(self):
        penalty = self.model.get_regularization_penalty()
        assert penalty == 0

        iterator = BasicIterator(batch_size=32)
        trainer = Trainer(self.model,
                          None,  # optimizer,
                          iterator,
                          self.instances)

        # You get a RuntimeError if you call `model.forward` twice on the same inputs.
        # The data and config are such that the whole dataset is one batch.
        training_batch = next(iterator(self.instances, num_epochs=1))
        validation_batch = next(iterator(self.instances, num_epochs=1))

        training_loss = trainer._batch_loss(training_batch, for_training=True).data
        validation_loss = trainer._batch_loss(validation_batch, for_training=False).data

        # Training loss should have the regularization penalty, but validation loss should not.
        assert (training_loss == validation_loss).all()


class SimpleTaggerRegularizationTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        param_file = 'tests/fixtures/simple_tagger/experiment_with_regularization.json'
        self.set_up_model(param_file,
                          'tests/fixtures/data/sequence_tagging.tsv')
        params = Params.from_file(param_file)
        self.reader = DatasetReader.from_params(params['dataset_reader'])
        self.iterator = DataIterator.from_params(params['iterator'])
        self.trainer = Trainer.from_params(
                self.model,
                self.TEST_DIR,
                self.iterator,
                self.dataset,
                None,
                params.get('trainer')
        )


    def test_regularization(self):
        penalty = self.model.get_regularization_penalty().data
        assert (penalty > 0).all()

        penalty2 = 0

        # Config specifies penalty as
        #   "regularizer": [
        #     ["weight$", {"type": "l2", "alpha": 10}],
        #     ["bias$", {"type": "l1", "alpha": 5}]
        #   ]
        for name, parameter in self.model.named_parameters():
            if name.endswith("weight"):
                weight_penalty = 10 * torch.sum(torch.pow(parameter, 2))
                penalty2 += weight_penalty
            elif name.endswith("bias"):
                bias_penalty = 5 * torch.sum(torch.abs(parameter))
                penalty2 += bias_penalty

        assert (penalty == penalty2.data).all()

        # You get a RuntimeError if you call `model.forward` twice on the same inputs.
        # The data and config are such that the whole dataset is one batch.
        training_batch = next(self.iterator(self.instances, num_epochs=1))
        validation_batch = next(self.iterator(self.instances, num_epochs=1))

        training_loss = self.trainer._batch_loss(training_batch, for_training=True).data
        validation_loss = self.trainer._batch_loss(validation_batch, for_training=False).data

        # Training loss should have the regularization penalty, but validation loss should not.
        assert (training_loss != validation_loss).all()

        # Training loss should equal the validation loss plus the penalty.
        penalized = validation_loss + penalty
        assert (training_loss == penalized).all()
