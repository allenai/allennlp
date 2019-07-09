import logging
import time
import datetime
from typing import Dict, Optional, List, Union, Any, Iterable

import torch

from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.callbacks.callback import Callback
from allennlp.training.callbacks.callback_handler import CallbackHandler
from allennlp.training.callbacks.events import Events
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("callback")
class CallbackTrainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 model_save_interval: Optional[float] = None,
                 cuda_device: Union[int, List] = -1,
                 callbacks: List[Callback] = None) -> None:
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
        num_epochs : int, optional (default=20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default=-1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        callbacks : ``List[Callback]``, optional (default=None)
            A list of callbacks that will be called based on training events.
        """
        super().__init__(serialization_dir, cuda_device)

        # This is all state that the callbacks might want:
        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model
        self.optimizer = optimizer
        self.validate = False

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

        self.training_batches: Iterable[List[TensorDict]] = ()
        self.num_training_batches = 0

        self.should_stop_early = False
        self.num_epochs = num_epochs

        self.training_start_time = 0.0
        self.checkpoint_epoch: Union[int, str] = 0
        self.model_save_interval = model_save_interval

        self.last_log = 0.0
        self.epoch_number = 0
        self.batch_grad_norm: Optional[float] = None
        self.handler = CallbackHandler(callbacks, self)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.

        This is a method on the trainer so that it can be used both in training and validation
        (which are handled separately).
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

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        self.handler.fire_event(Events.RESTORE_CHECKPOINT)
        starting_epoch = self.epoch_number

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

            batch_groups_tqdm = Tqdm.tqdm(self.training_batches, total=self.num_training_batches)

            for self.batch_group in batch_groups_tqdm:
                self.handler.fire_event(Events.BATCH_START)

                self.batches_this_epoch += 1
                self.batch_num_total += 1

                self.handler.fire_event(Events.FORWARD)
                self.handler.fire_event(Events.BACKWARD)

                description = training_util.description_from_metrics(self.train_metrics)

                batch_groups_tqdm.set_description(description, refresh=False)

                # Save model if needed.
                if self.model_save_interval is not None and (
                        time.time() - last_save_time > self.model_save_interval
                ):
                    last_save_time = time.time()
                    self.checkpoint_epoch = f"{self.epoch_number}.{training_util.time_to_str(int(last_save_time))}"
                    self.handler.fire_event(Events.SAVE_CHECKPOINT)

                self.handler.fire_event(Events.BATCH_END)

            self.handler.fire_event(Events.VALIDATE)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if self.epoch_number < self.num_epochs - 1:
                training_elapsed_time = time.time() - self.training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self.num_epochs - starting_epoch) / float(self.epoch_number - starting_epoch + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            self.handler.fire_event(Events.EPOCH_END)

            self.checkpoint_epoch = self.epoch_number
            self.handler.fire_event(Events.SAVE_CHECKPOINT)

            if self.should_stop_early:
                logger.info("Ran out of patience.  Stopping training.")
                break

        self.handler.fire_event(Events.TRAINING_END)

        return self.metrics

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'CallbackTrainer':
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        model = pieces.model
        params = pieces.params
        validation_iterator = pieces.validation_iterator or pieces.iterator

        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))

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
                                                          instances=pieces.train_dataset,
                                                          iterator=pieces.iterator,
                                                          shuffle=shuffle,
                                                          validation_data=pieces.validation_dataset,
                                                          validation_iterator=validation_iterator,
                                                          serialization_dir=serialization_dir)
                                     for callback_params in callbacks_params]

        params.assert_empty(cls.__name__)
        return cls(model, optimizer,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   model_save_interval=model_save_interval,
                   callbacks=callbacks)
