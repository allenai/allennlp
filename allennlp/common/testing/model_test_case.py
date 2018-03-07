import copy
import os

from numpy.testing import assert_allclose
import torch

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive


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
        instances = reader.read(dataset_file)
        vocab = Vocabulary.from_instances(instances)
        self.vocab = vocab
        self.instances = instances
        self.model = Model.from_params(self.vocab, params['model'])

        # TODO(joelgrus) get rid of these
        # (a lot of the model tests use them, so they'll have to be changed)
        self.dataset = Batch(self.instances)
        self.dataset.index_instances(self.vocab)

    def ensure_model_can_train_save_and_load(self,
                                             param_file: str,
                                             tolerance: float = 1e-4,
                                             cuda_device: int = -1):
        save_dir = os.path.join(self.TEST_DIR, "save_and_load_test")
        archive_file = os.path.join(save_dir, "model.tar.gz")
        model = train_model_from_file(param_file, save_dir)
        loaded_model = load_archive(archive_file, cuda_device=cuda_device).model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # First we make sure that the state dict (the parameters) are the same for both models.
        for key in state_keys:
            assert_allclose(model.state_dict()[key].cpu().numpy(),
                            loaded_model.state_dict()[key].cpu().numpy(),
                            err_msg=key)
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])

        # Need to duplicate params because Iterator.from_params will consume.
        iterator_params = params['iterator']
        iterator_params2 = Params(copy.deepcopy(iterator_params.as_dict()))

        iterator = DataIterator.from_params(iterator_params)
        iterator2 = DataIterator.from_params(iterator_params2)

        # We'll check that even if we index the dataset with each model separately, we still get
        # the same result out.
        model_dataset = reader.read(params['validation_data_path'])
        iterator.index_with(model.vocab)
        model_batch = next(iterator(model_dataset, shuffle=False, cuda_device=cuda_device))

        loaded_dataset = reader.read(params['validation_data_path'])
        iterator2.index_with(loaded_model.vocab)
        loaded_batch = next(iterator2(loaded_dataset, shuffle=False, cuda_device=cuda_device))

        # Check gradients are None for non-trainable parameters and check that
        # trainable parameters receive some gradient if they are trainable.
        self.check_model_computes_gradients_correctly(model, model_batch)

        # The datasets themselves should be identical.
        assert model_batch.keys() == loaded_batch.keys()
        for key in model_batch.keys():
            self.assert_fields_equal(model_batch[key], loaded_batch[key], key, 1e-6)

        # Set eval mode, to turn off things like dropout, then get predictions.
        model.eval()
        loaded_model.eval()
        # Models with stateful RNNs need their states reset to have consistent
        # behavior after loading.
        for model_ in [model, loaded_model]:
            for module in model_.modules():
                if hasattr(module, 'stateful') and module.stateful:
                    module.reset_states()
        model_predictions = model(**model_batch)
        loaded_model_predictions = loaded_model(**loaded_batch)

        # Check loaded model's loss exists and we can compute gradients, for continuing training.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values for these keys should be close.
        for key in model_predictions.keys():
            self.assert_fields_equal(model_predictions[key],
                                     loaded_model_predictions[key],
                                     name=key,
                                     tolerance=tolerance)

        return model, loaded_model

    def assert_fields_equal(self, field1, field2, name: str, tolerance: float = 1e-6) -> None:
        if isinstance(field1, torch.autograd.Variable):
            assert_allclose(field1.data.cpu().numpy(),
                            field2.data.cpu().numpy(),
                            rtol=tolerance,
                            err_msg=name)
        elif isinstance(field1, dict):
            assert field1.keys() == field2.keys()
            for key in field1:
                self.assert_fields_equal(field1[key],
                                         field2[key],
                                         tolerance=tolerance,
                                         name=name + '.' + key)
        elif isinstance(field1, (list, tuple)):
            assert len(field1) == len(field2)
            for i, (subfield1, subfield2) in enumerate(zip(field1, field2)):
                self.assert_fields_equal(subfield1,
                                         subfield2,
                                         tolerance=tolerance,
                                         name=name + f"[{i}]")
        else:
            assert field1 == field2

    @staticmethod
    def check_model_computes_gradients_correctly(model, model_batch):
        model.zero_grad()
        result = model(**model_batch)
        result["loss"].backward()

        for parameter in model.parameters():
            zeros = torch.zeros(parameter.size())
            if parameter.requires_grad:
                # Some parameters will only be partially updated,
                # like embeddings, so we just check that any gradient is non-zero.
                assert (parameter.grad.data.cpu() != zeros).any()
            else:
                assert parameter.grad is None

    def ensure_batch_predictions_are_consistent(self):
        self.model.eval()
        single_predictions = []
        for i, instance in enumerate(self.instances):
            dataset = Batch([instance])
            tensors = dataset.as_tensor_dict(dataset.get_padding_lengths(), for_training=False)
            result = self.model(**tensors)
            single_predictions.append(result)
        full_dataset = Batch(self.instances)
        batch_tensors = full_dataset.as_tensor_dict(full_dataset.get_padding_lengths(), for_training=False)
        batch_predictions = self.model(**batch_tensors)
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
                        slices = tuple(slice(0, size) for size in single_predicted.size())
                        batch_predicted = batch_predicted[slices]
                    assert_allclose(single_predicted.data.numpy(),
                                    batch_predicted.data.numpy(),
                                    atol=tolerance,
                                    err_msg=key)
                else:
                    assert single_predicted == batch_predicted, key
