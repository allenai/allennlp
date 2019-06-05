import logging
import math
import time
import datetime
from typing import Dict, Optional, List, Union, Any, Iterable

import torch

from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.callbacks import Callback, CallbackHandler, Events
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("trainer2")
class Trainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 callbacks: List[Callback['Trainer']] = None) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        """
        super().__init__(serialization_dir, cuda_device)

        # This is all state that the callbacks might want:
        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model
        self.iterator = iterator
        self.validation_iterator = validation_iterator or iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        # For capturing mid / end-of-epoch metrics
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.latest_val_metric = 0.0
        self.train_loss = 0.0

        # For capturing overall metrics
        self.metrics: Dict[str, Any] = {}

        self.batch_num_total = 0
        self.batch_group: List[TensorDict] = []
        self.batches_this_epoch = 0
        self.should_stop_early = False
        self.num_epochs = num_epochs

        self.training_start_time = 0.0
        self.checkpoint_epoch: Union[int, str] = 0
        self.model_save_interval = model_save_interval

        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping
        self.last_log = 0.0
        self.epoch_number = 0
        self.batch_grad_norm: Optional[float] = None
        self.handler = CallbackHandler(callbacks, self)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def generate_batch_groups(self) -> Iterable[List[TensorDict]]:
        """
        Returns an iterable over the batch groups
        """
        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        return train_generator

    def num_training_batches(self) -> int:
        num_gpus = len(self._cuda_devices)
        return math.ceil(self.iterator.get_num_batches(self.train_data) / num_gpus)

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        self.handler.fire_event(Events.RESTORE_CHECKPOINT)
        starting_epoch = self.epoch_number

        training_util.enable_gradient_clipping(self.model, self.grad_clipping)

        logger.info("Beginning training.")
        self.handler.fire_event(Events.TRAINING_START)

        self.training_start_time = time.time()

        for self.epoch_number in range(starting_epoch, self.num_epochs):
            epoch_start_time = time.time()
            ####
            self.handler.fire_event(Events.EPOCH_START)

            self.train_loss = 0.0
            # Set the model to "train" mode.
            self.model.train()

            self.last_log = time.time()
            last_save_time = time.time()

            logger.info("Training")
            self.batches_this_epoch = 0
            batch_groups_tqdm = Tqdm.tqdm(self.generate_batch_groups(),
                                          total=self.num_training_batches())

            for self.batch_group in batch_groups_tqdm:
                self.handler.fire_event(Events.BATCH_START)

                self.batches_this_epoch += 1
                self.batch_num_total += 1

                self.optimizer.zero_grad()
                loss = self.batch_loss(self.batch_group, for_training=True)

                ####
                self.handler.fire_event(Events.AFTER_FORWARD)

                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")

                loss.backward()

                ####
                self.handler.fire_event(Events.AFTER_BACKWARD)

                self.train_loss += loss.item()

                self.batch_grad_norm = training_util.rescale_gradients(self.model, self.grad_norm)

                self.optimizer.step()

                # Update the description with the latest metrics
                self.train_metrics.update(
                        training_util.get_metrics(self.model, self.train_loss, self.batches_this_epoch)
                )
                description = training_util.description_from_metrics(self.train_metrics)

                batch_groups_tqdm.set_description(description, refresh=False)

                # Save model if needed.
                if self.model_save_interval is not None and (
                        time.time() - last_save_time > self.model_save_interval
                ):
                    last_save_time = time.time()
                    self.checkpoint_epoch = f"{self.epoch_number}.{training_util.time_to_str(int(last_save_time))}"
                    self.handler.fire_sequence(Events.SAVE_CHECKPOINT)

                ####
                self.handler.fire_event(Events.BATCH_END)

            ####
            self.handler.fire_sequence(Events.VALIDATE)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if self.epoch_number < self.num_epochs - 1:
                training_elapsed_time = time.time() - self.training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self.num_epochs - starting_epoch) / float(self.epoch_number - starting_epoch + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            ####
            self.handler.fire_event(Events.EPOCH_END)

            self.checkpoint_epoch = self.epoch_number

            self.handler.fire_sequence(Events.SAVE_CHECKPOINT)

            if self.should_stop_early:
                logger.info("Ran out of patience.  Stopping training.")
                break


        ####
        self.handler.fire_event(Events.TRAINING_END)

        return self.metrics

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'Trainer':
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        model = pieces.model
        iterator = pieces.iterator
        train_data = pieces.train_dataset
        validation_data = pieces.validation_dataset
        params = pieces.params
        validation_iterator = pieces.validation_iterator

        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        model_save_interval = params.pop_float("model_save_interval", None)

        callbacks_params = params.pop("callbacks", [])
        callbacks: List[Callback] = [Callback.from_params(params=callback_params,
                                                          model=model,
                                                          optimizer=optimizer,
                                                          serialization_dir=serialization_dir)
                                     for callback_params in callbacks_params]

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   model_save_interval=model_save_interval,
                   callbacks=callbacks)
