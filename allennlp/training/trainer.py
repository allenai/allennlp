"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""
# pylint: disable=too-many-lines

import logging
import os
from typing import Dict, Optional, List, Union, Any

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather
from tensorboardX import SummaryWriter

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import scatter_kwargs
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.util import sparse_clip_norm

logger = logging.getLogger(__name__)


class Trainer(Registrable):
    """
    The base class for an AllenNLP trainer.
    The only method you need to implement is ``train``.
    """
    default_implementation = "default"

    def __init__(self,
                 model: Model,
                 serialization_dir: str,
                 cuda_device: Union[int, List] = -1) -> None:
        check_for_gpu(cuda_device)

        self.model = model
        self._serialization_dir = serialization_dir

        self._set_up_tensorboard()
        self._configure_gpus(cuda_device)
        self._warned_tqdm_ignores_underscores = False

    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    def batch_loss(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            batch = util.move_to_device(batch, self._cuda_devices[0])
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

    def _configure_gpus(self, cuda_device: Union[int, List]) -> None:
        if not isinstance(cuda_device, int) and not isinstance(cuda_device, list):
            raise ConfigurationError("Expected an int or list for cuda_device, got {}".format(cuda_device))

        if isinstance(cuda_device, list):
            logger.warning(f"Multiple GPU support is experimental not recommended for use. "
                           "In some cases it may lead to incorrect results or undefined behavior.")
            self._multiple_gpu = True
            self._cuda_devices = cuda_device
        else:
            self._multiple_gpu = False
            self._cuda_devices = [cuda_device]

        if self._cuda_devices[0] != -1:
            self.model = self.model.cuda(self._cuda_devices[0])

    def _set_up_tensorboard(self) -> None:
        if self._serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "validation"))
            self._tensorboard = TensorboardWriter(train_log, validation_log)
        else:
            self._tensorboard = TensorboardWriter()

    def _enable_gradient_clipping(self, grad_clipping: Optional[float]) -> None:
        if grad_clipping is not None:
            clip_function = lambda grad: grad.clamp(-grad_clipping, grad_clipping)
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)


    def _rescale_gradients(self, grad_norm: Optional[float] = None) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if grad_norm:
            parameters_to_clip = [p for p in self.model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, grad_norm)
        return None

    def _get_metrics(self, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``num_batches`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self.model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
        return metrics

    def _get_batch_size(self, batch: Union[Dict, torch.Tensor]) -> int:
        """
        Returns the size of the batch dimension. Assumes a well-formed batch,
        returns 0 otherwise.
        """
        if isinstance(batch, torch.Tensor):
            return batch.size(0) # type: ignore
        elif isinstance(batch, Dict):
            return self._get_batch_size(next(iter(batch.values())))
        else:
            return 0

    def _data_parallel(self, batch):
        """
        Do the forward pass using multiple GPUs.  This is a simplification
        of torch.nn.parallel.data_parallel to support the allennlp model
        interface.
        """
        inputs, module_kwargs = scatter_kwargs((), batch, self._cuda_devices, 0)

        used_device_ids = self._cuda_devices[:len(inputs)]
        replicas = replicate(self.model, used_device_ids)
        outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)

        # Only the 'loss' is needed.
        # a (num_gpu, ) tensor with loss on each GPU
        losses = gather([output['loss'].unsqueeze(0) for output in outputs], used_device_ids[0], 0)
        return {'loss': losses.mean()}

    @classmethod
    def from_params(cls,   # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False):
        # pylint: disable=arguments-differ
        typ3 = params.pop("type", cls.default_implementation)
        return Trainer.by_name(typ3).from_params(params, serialization_dir, recover)
