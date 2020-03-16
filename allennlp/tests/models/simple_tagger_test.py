from flaky import flaky
import numpy
import pytest
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import DataLoader
from allennlp.models import Model
from allennlp.training import GradientDescentTrainer, Trainer


class SimpleTaggerTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "simple_tagger" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv",
        )

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.make_output_human_readable(output_dict)
        class_probs = output_dict["class_probabilities"][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1), numpy.array([1, 1, 1, 1]))

    def test_forward_on_instances_ignores_loss_key_when_batched(self):
        batch_outputs = self.model.forward_on_instances(self.dataset.instances)
        for output in batch_outputs:
            assert "loss" not in output.keys()

        # It should be in the single batch case, because we special case it.
        single_output = self.model.forward_on_instance(self.dataset.instances[0])
        assert "loss" in single_output.keys()

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_regularization(self):
        penalty = self.model.get_regularization_penalty()
        assert penalty == 0

        data_loader = DataLoader(self.instances, batch_size=32)
        trainer = GradientDescentTrainer(self.model, None, data_loader)  # optimizer,

        # You get a RuntimeError if you call `model.forward` twice on the same inputs.
        # The data and config are such that the whole dataset is one batch.
        training_batch = next(iter(data_loader))
        validation_batch = next(iter(data_loader))

        training_loss = trainer.batch_loss(training_batch, for_training=True).item()
        validation_loss = trainer.batch_loss(validation_batch, for_training=False).item()

        # Training loss should have the regularization penalty, but validation loss should not.
        numpy.testing.assert_almost_equal(training_loss, validation_loss)


class SimpleTaggerSpanF1Test(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "simple_tagger_with_span_f1" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "conll2003.txt",
        )

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()


class SimpleTaggerRegularizationTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        param_file = self.FIXTURES_ROOT / "simple_tagger" / "experiment_with_regularization.json"
        self.set_up_model(param_file, self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv")
        params = Params.from_file(param_file)
        self.reader = DatasetReader.from_params(params["dataset_reader"])
        self.data_loader = DataLoader.from_params(
            dataset=self.instances, params=params["data_loader"]
        )
        self.trainer = Trainer.from_params(
            model=self.model,
            data_loader=self.data_loader,
            serialization_dir=self.TEST_DIR,
            params=params.get("trainer"),
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
        training_batch = next(iter(self.data_loader))
        validation_batch = next(iter(self.data_loader))

        training_loss = self.trainer.batch_loss(training_batch, for_training=True).data
        validation_loss = self.trainer.batch_loss(validation_batch, for_training=False).data

        # Training loss should have the regularization penalty, but validation loss should not.
        assert (training_loss != validation_loss).all()

        # Training loss should equal the validation loss plus the penalty.
        penalized = validation_loss + penalty
        assert (training_loss == penalized).all()
