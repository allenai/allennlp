from collections import defaultdict
import inspect
from typing import Dict, List

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.modules import Backbone
from allennlp.models.model import Model
from allennlp.models.heads import Head
from allennlp.nn import InitializerApplicator



# TODO:
# 1. implement get_metrics
# 2. implement make_output_human_readable
# 3. implement selective arguments / which heads to run at training vs. eval, etc.


def get_forward_arguments(module: torch.nn.Module) -> List[str]:
    signature = inspect.signature(module.forward)
    arguments = []
    for arg in signature.parameters:
        if arg == "self":
            continue
        arguments.append(arg)
    return arguments


class MultiTaskModel(Model):
    """
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """
    def __init__(
        self,
        vocab: Vocabulary,
        backbone: Backbone,
        heads: Dict[str, Head],
        *,
        arg_name_mapping: Dict[str, str] = None,
        backbone_arguments: List[str] = None,
        head_arguments: Dict[str, List[str]] = None,
        loss_weights: Dict[str, float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ):
        super().__init__(vocab, **kwargs)
        self._backbone = backbone
        self._heads = heads
        self._arg_name_mapping = arg_name_mapping or {}

        self._backbone_arguments = backbone_arguments or get_forward_arguments(backbone)
        self._head_arguments = head_arguments or {
            key: get_forward_arguments(heads[key]) for key in heads
        }
        self._loss_weights = loss_weights or defaultdict(lambda: 1.0)
        initializer(self)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        backbone_arguments = {}
        for key, value in kwargs.items():
            new_key = self._arg_name_mapping.get(key, key)
            if new_key in self._backbone_arguments:
                backbone_arguments[new_key] = value

        backbone_outputs = self._backbone(**backbone_arguments)

        outputs = {}
        loss = None
        for head_name in self._heads:
            combined_arguments = {**backbone_outputs, **kwargs}
            head_arguments = {}
            for key, value in combined_arguments.items():
                # This is similar to the name replacement logic above, but it can't be combined,
                # because we don't want to clobber values if, e.g., we have two different classifier
                # heads that both need to use the "label" key.  This must be done separately for
                # each head.
                new_key = self._arg_name_mapping.get(key, key)
                if new_key in self._head_arguments[head_name]:
                    head_arguments[new_key] = value

            head_outputs = self._heads[head_name](**head_arguments)
            for key in head_outputs:
                outputs[f"{head_name}_{key}"] = head_outputs[key]

            if "loss" in head_outputs:
                head_loss = self._loss_weights[head_name] * head_outputs["loss"]
                if loss is None:
                    loss = head_loss
                else:
                    loss += head_loss

        if loss is not None:
            outputs["loss"] = loss

        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
