import os

from numpy.testing import assert_allclose
import torch

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DataIterator, Dataset, DatasetReader, Vocabulary
from allennlp.models import Model, load_archive
from allennlp.nn.util import arrays_to_variables


class ModelTestCase(AllenNlpTestCase):
    """
    A subclass of :class:`~allennlp.common.testing.test_case.AllenNlpTestCase`
    with added methods for testing :class:`~allennlp.models.model.Model` subclasses.
    """
    def set_up_model(self, param_file, dataset_file):
        # pylint: disable=attribute-defined-outside-init
        self.param_file = param_file
        params = Params.from_file(self.param_file)

        reader = DatasetReader.from_params(params['dataset_reader'])
        dataset = reader.read(dataset_file)
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.model = Model.from_params(self.vocab, params['model'])

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
                    self.assert_fields_equal(model_batch[key][subfield],
                                             loaded_batch[key][subfield],
                                             tolerance=1e-6,
                                             name=key + '.' + subfield)
            else:
                self.assert_fields_equal(model_batch[key], loaded_batch[key], 1e-6, key)

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
            self.assert_fields_equal(model_predictions[key],
                                     loaded_model_predictions[key],
                                     tolerance=1e-4,
                                     name=key)

        return model, loaded_model

    @staticmethod
    def assert_fields_equal(field1, field2, tolerance: float = 1e-6, name: str = None) -> None:
        if isinstance(field1, torch.autograd.Variable):
            assert_allclose(field1.data.numpy(),
                            field2.data.numpy(),
                            rtol=tolerance,
                            err_msg=name)
        else:
            assert field1 == field2

    def ensure_batch_predictions_are_consistent(self):
        self.model.eval()
        single_predictions = []
        for i, instance in enumerate(self.dataset.instances):
            dataset = Dataset([instance])
            arrays = dataset.as_array_dict(dataset.get_padding_lengths(), verbose=False)
            variables = arrays_to_variables(arrays, for_training=False)
            result = self.model.forward(**variables)
            single_predictions.append(result)
        batch_arrays = self.dataset.as_array_dict(self.dataset.get_padding_lengths(), verbose=False)
        batch_variables = arrays_to_variables(batch_arrays, for_training=False)
        batch_predictions = self.model.forward(**batch_variables)
        for i, instance_predictions in enumerate(single_predictions):
            for key, single_predicted in instance_predictions.items():
                tolerance = 1e-6
                if key == 'loss':
                    # Loss is particularly unstable; we'll just be satisfied if everything else is
                    # close.
                    continue
                single_predicted = single_predicted[0]
                batch_predicted = batch_predictions[key][i]
                if isinstance(single_predicted, torch.autograd.Variable):
                    if single_predicted.size() != batch_predicted.size():
                        # This is probably a sequence model, and our output shape has some padded
                        # elements in the batched case.  Fixing this in general is complicated;
                        # we'll just fix some easy cases that we actually have, for now.
                        num_tokens = single_predicted.size(0)
                        if batch_predicted.dim() == 1:
                            batch_predicted = batch_predicted[:num_tokens]
                        elif batch_predicted.dim() == 2:
                            batch_predicted = batch_predicted[:num_tokens, :]
                        else:
                            raise NotImplementedError
                    assert_allclose(single_predicted.data.numpy(),
                                    batch_predicted.data.numpy(),
                                    atol=tolerance,
                                    err_msg=key)
                else:
                    assert single_predicted == batch_predicted, key
