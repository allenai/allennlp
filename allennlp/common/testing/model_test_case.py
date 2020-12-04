import copy
import json
from os import PathLike
import random
from typing import Any, Dict, Iterable, Set, Union

import torch
import numpy
from numpy.testing import assert_allclose

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.data.batch import Batch
from allennlp.models import load_archive, Model
from allennlp.training import GradientDescentTrainer


class ModelTestCase(AllenNlpTestCase):
    """
    A subclass of [`AllenNlpTestCase`](./test_case.md)
    with added methods for testing [`Model`](../../models/model.md) subclasses.
    """

    def set_up_model(
        self,
        param_file: PathLike,
        dataset_file: PathLike,
        serialization_dir: PathLike = None,
        seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

        self.param_file = str(param_file)
        params = Params.from_file(self.param_file)

        reader = DatasetReader.from_params(
            params["dataset_reader"], serialization_dir=serialization_dir
        )
        # The dataset reader might be lazy, but a lazy list here breaks some of our tests.
        instances = list(reader.read(str(dataset_file)))
        # Use parameters for vocabulary if they are present in the config file, so that choices like
        # "non_padded_namespaces", "min_count" etc. can be set if needed.
        if "vocabulary" in params:
            vocab_params = params["vocabulary"]
            vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
        else:
            vocab = Vocabulary.from_instances(instances)
        self.vocab = vocab
        self.instances = instances
        self.model = Model.from_params(
            vocab=self.vocab, params=params["model"], serialization_dir=serialization_dir
        )

        # TODO(joelgrus) get rid of these
        # (a lot of the model tests use them, so they'll have to be changed)
        self.dataset = Batch(self.instances)
        self.dataset.index_instances(self.vocab)

    def ensure_model_can_train_save_and_load(
        self,
        param_file: Union[PathLike, str],
        tolerance: float = 1e-4,
        cuda_device: int = -1,
        gradients_to_ignore: Set[str] = None,
        overrides: str = "",
        metric_to_check: str = None,
        metric_terminal_value: float = None,
        metric_tolerance: float = 1e-4,
        disable_dropout: bool = True,
        seed: int = None,
    ):
        """
        # Parameters

        param_file : `str`
            Path to a training configuration file that we will use to train the model for this
            test.
        tolerance : `float`, optional (default=`1e-4`)
            When comparing model predictions between the originally-trained model and the model
            after saving and loading, we will use this tolerance value (passed as `rtol` to
            `numpy.testing.assert_allclose`).
        cuda_device : `int`, optional (default=`-1`)
            The device to run the test on.
        gradients_to_ignore : `Set[str]`, optional (default=`None`)
            This test runs a gradient check to make sure that we're actually computing gradients
            for all of the parameters in the model.  If you really want to ignore certain
            parameters when doing that check, you can pass their names here.  This is not
            recommended unless you're `really` sure you don't need to have non-zero gradients for
            those parameters (e.g., some of the beam search / state machine models have
            infrequently-used parameters that are hard to force the model to use in a small test).
        overrides : `str`, optional (default = `""`)
            A JSON string that we will use to override values in the input parameter file.
        metric_to_check: `str`, optional (default = `None`)
            We may want to automatically perform a check that model reaches given metric when
            training (on validation set, if it is specified). It may be useful in CI, for example.
            You can pass any metric that is in your model returned metrics.
        metric_terminal_value: `str`, optional (default = `None`)
            When you set `metric_to_check`, you need to set the value this metric must converge to
        metric_tolerance: `float`, optional (default=`1e-4`)
            Tolerance to check you model metric against metric terminal value. One can expect some
            variance in model metrics when the training process is highly stochastic.
        disable_dropout : `bool`, optional (default = `True`)
            If True we will set all dropout to 0 before checking gradients. (Otherwise, with small
            datasets, you may get zero gradients because of unlucky dropout.)
        """
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        model = train_model_from_file(param_file, save_dir, overrides=overrides)
        assert model is not None
        metrics_file = save_dir / "metrics.json"
        if metric_to_check is not None:
            metrics = json.loads(metrics_file.read_text())
            metric_value = metrics.get(f"best_validation_{metric_to_check}") or metrics.get(
                f"training_{metric_to_check}"
            )
            assert metric_value is not None, f"Cannot find {metric_to_check} in metrics.json file"
            assert metric_terminal_value is not None, "Please specify metric terminal value"
            assert abs(metric_value - metric_terminal_value) < metric_tolerance
        archive = load_archive(archive_file, cuda_device=cuda_device)
        loaded_model = archive.model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # First we make sure that the state dict (the parameters) are the same for both models.
        for key in state_keys:
            assert_allclose(
                model.state_dict()[key].cpu().numpy(),
                loaded_model.state_dict()[key].cpu().numpy(),
                err_msg=key,
            )
        reader = archive.dataset_reader
        params = Params.from_file(param_file, params_overrides=overrides)

        # Need to duplicate params because DataLoader.from_params will consume.
        data_loader_params = params["data_loader"]
        data_loader_params["shuffle"] = False
        data_loader_params2 = Params(copy.deepcopy(data_loader_params.as_dict()))

        print("Reading with original model")
        data_loader = DataLoader.from_params(
            params=data_loader_params, reader=reader, data_path=params["validation_data_path"]
        )
        data_loader.index_with(model.vocab)

        print("Reading with loaded model")
        data_loader2 = DataLoader.from_params(
            params=data_loader_params2, reader=reader, data_path=params["validation_data_path"]
        )
        data_loader2.index_with(loaded_model.vocab)

        # We'll check that even if we index the dataset with each model separately, we still get
        # the same result out.
        model_batch = next(iter(data_loader))

        loaded_batch = next(iter(data_loader2))

        # Check gradients are None for non-trainable parameters and check that
        # trainable parameters receive some gradient if they are trainable.
        self.check_model_computes_gradients_correctly(
            model, model_batch, gradients_to_ignore, disable_dropout
        )

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
                if hasattr(module, "stateful") and module.stateful:
                    module.reset_states()
        print("Predicting with original model")
        model_predictions = model(**model_batch)
        print("Predicting with loaded model")
        loaded_model_predictions = loaded_model(**loaded_batch)

        # Both outputs should have the same keys and the values for these keys should be close.
        for key in model_predictions.keys():
            self.assert_fields_equal(
                model_predictions[key], loaded_model_predictions[key], name=key, tolerance=tolerance
            )

        # Check loaded model's loss exists and we can compute gradients, for continuing training.
        loaded_model.train()
        loaded_model_predictions = loaded_model(**loaded_batch)
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        return model, loaded_model

    def ensure_model_can_train(
        self,
        trainer: GradientDescentTrainer,
        gradients_to_ignore: Set[str] = None,
        metric_to_check: str = None,
        metric_terminal_value: float = None,
        metric_tolerance: float = 1e-4,
        disable_dropout: bool = True,
    ):
        """
        A simple test for model training behavior when you are not using configuration files. In
        this case, we don't have a story around saving and loading models (you need to handle that
        yourself), so we don't have tests for that.  We just test that the model can train, and that
        it computes gradients for all parameters.

        Because the `Trainer` already has a reference to a model and to a data loader, we just take
        the `Trainer` object itself, and grab the `Model` and other necessary objects from there.

        # Parameters

        trainer: `GradientDescentTrainer`
            The `Trainer` to use for the test, which already has references to a `Model` and a
            `DataLoader`, which we will use in the test.
        gradients_to_ignore : `Set[str]`, optional (default=`None`)
            This test runs a gradient check to make sure that we're actually computing gradients
            for all of the parameters in the model.  If you really want to ignore certain
            parameters when doing that check, you can pass their names here.  This is not
            recommended unless you're `really` sure you don't need to have non-zero gradients for
            those parameters (e.g., some of the beam search / state machine models have
            infrequently-used parameters that are hard to force the model to use in a small test).
        metric_to_check: `str`, optional (default = `None`)
            We may want to automatically perform a check that model reaches given metric when
            training (on validation set, if it is specified). It may be useful in CI, for example.
            You can pass any metric that is in your model returned metrics.
        metric_terminal_value: `str`, optional (default = `None`)
            When you set `metric_to_check`, you need to set the value this metric must converge to
        metric_tolerance: `float`, optional (default=`1e-4`)
            Tolerance to check you model metric against metric terminal value. One can expect some
            variance in model metrics when the training process is highly stochastic.
        disable_dropout : `bool`, optional (default = `True`)
            If True we will set all dropout to 0 before checking gradients. (Otherwise, with small
            datasets, you may get zero gradients because of unlucky dropout.)
        """
        metrics = trainer.train()
        if metric_to_check is not None:
            metric_value = metrics.get(f"best_validation_{metric_to_check}") or metrics.get(
                f"training_{metric_to_check}"
            )
            assert metric_value is not None, f"Cannot find {metric_to_check} in metrics.json file"
            assert metric_terminal_value is not None, "Please specify metric terminal value"
            assert abs(metric_value - metric_terminal_value) < metric_tolerance

        model_batch = next(iter(trainer.data_loader))

        # Check gradients are None for non-trainable parameters and check that
        # trainable parameters receive some gradient if they are trainable.
        self.check_model_computes_gradients_correctly(
            trainer.model, model_batch, gradients_to_ignore, disable_dropout
        )

    def assert_fields_equal(self, field1, field2, name: str, tolerance: float = 1e-6) -> None:
        if isinstance(field1, torch.Tensor):
            assert_allclose(
                field1.detach().cpu().numpy(),
                field2.detach().cpu().numpy(),
                rtol=tolerance,
                err_msg=name,
            )
        elif isinstance(field1, dict):
            assert field1.keys() == field2.keys()
            for key in field1:
                self.assert_fields_equal(
                    field1[key], field2[key], tolerance=tolerance, name=name + "." + str(key)
                )
        elif isinstance(field1, (list, tuple)):
            assert len(field1) == len(field2)
            for i, (subfield1, subfield2) in enumerate(zip(field1, field2)):
                self.assert_fields_equal(
                    subfield1, subfield2, tolerance=tolerance, name=name + f"[{i}]"
                )
        elif isinstance(field1, (float, int)):
            assert_allclose([field1], [field2], rtol=tolerance, err_msg=name)
        else:
            if field1 != field2:
                for key in field1.__dict__:
                    print(key, getattr(field1, key) == getattr(field2, key))
            assert field1 == field2, f"{name}, {type(field1)}, {type(field2)}"

    @staticmethod
    def check_model_computes_gradients_correctly(
        model: Model,
        model_batch: Dict[str, Union[Any, Dict[str, Any]]],
        params_to_ignore: Set[str] = None,
        disable_dropout: bool = True,
    ):
        print("Checking gradients")
        for p in model.parameters():
            p.grad = None
        model.train()

        original_dropouts: Dict[str, float] = {}

        if disable_dropout:
            # Remember original dropouts so we can restore them.
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    original_dropouts[name] = getattr(module, "p")
                    setattr(module, "p", 0)

        result = model(**model_batch)
        result["loss"].backward()
        has_zero_or_none_grads = {}
        for name, parameter in model.named_parameters():
            zeros = torch.zeros(parameter.size())
            if params_to_ignore and name in params_to_ignore:
                continue
            if parameter.requires_grad:

                if parameter.grad is None:
                    has_zero_or_none_grads[
                        name
                    ] = "No gradient computed (i.e parameter.grad is None)"

                elif parameter.grad.is_sparse or parameter.grad.data.is_sparse:
                    pass

                # Some parameters will only be partially updated,
                # like embeddings, so we just check that any gradient is non-zero.
                elif (parameter.grad.cpu() == zeros).all():
                    has_zero_or_none_grads[
                        name
                    ] = f"zeros with shape ({tuple(parameter.grad.size())})"
            else:
                assert parameter.grad is None

        if has_zero_or_none_grads:
            for name, grad in has_zero_or_none_grads.items():
                print(f"Parameter: {name} had incorrect gradient: {grad}")
            raise Exception("Incorrect gradients found. See stdout for more info.")

        # Now restore dropouts if we disabled them.
        if disable_dropout:
            for name, module in model.named_modules():
                if name in original_dropouts:
                    setattr(module, "p", original_dropouts[name])

    def ensure_batch_predictions_are_consistent(self, keys_to_ignore: Iterable[str] = ()):
        """
        Ensures that the model performs the same on a batch of instances as on individual instances.
        Ignores metrics matching the regexp .*loss.* and those specified explicitly.

        # Parameters

        keys_to_ignore : `Iterable[str]`, optional (default=`()`)
            Names of metrics that should not be taken into account, e.g. "batch_weight".
        """
        self.model.eval()
        single_predictions = []
        for i, instance in enumerate(self.instances):
            dataset = Batch([instance])
            tensors = dataset.as_tensor_dict(dataset.get_padding_lengths())
            result = self.model(**tensors)
            single_predictions.append(result)
        full_dataset = Batch(self.instances)
        batch_tensors = full_dataset.as_tensor_dict(full_dataset.get_padding_lengths())
        batch_predictions = self.model(**batch_tensors)
        for i, instance_predictions in enumerate(single_predictions):
            for key, single_predicted in instance_predictions.items():
                tolerance = 1e-6
                if "loss" in key:
                    # Loss is particularly unstable; we'll just be satisfied if everything else is
                    # close.
                    continue
                if key in keys_to_ignore:
                    continue
                single_predicted = single_predicted[0]
                batch_predicted = batch_predictions[key][i]
                if isinstance(single_predicted, torch.Tensor):
                    if single_predicted.size() != batch_predicted.size():
                        slices = tuple(slice(0, size) for size in single_predicted.size())
                        batch_predicted = batch_predicted[slices]
                    assert_allclose(
                        single_predicted.data.numpy(),
                        batch_predicted.data.numpy(),
                        atol=tolerance,
                        err_msg=key,
                    )
                else:
                    assert single_predicted == batch_predicted, key
