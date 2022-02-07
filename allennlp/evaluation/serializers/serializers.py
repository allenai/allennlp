from typing import Optional, Dict, Any, Callable
import logging
import json

from allennlp.common.util import sanitize
from allennlp.data.fields import TensorField
from allennlp.common import Registrable
from allennlp.data import DataLoader

logger = logging.getLogger(__name__)


class Serializer(Registrable):
    """
    General serializer class for turning batches into human readable data
    """

    def __call__(
        self,
        batch: Dict[str, TensorField],
        output_dict: Dict,
        data_loader: DataLoader,
        output_postprocess_function: Optional[Callable] = None,
    ) -> str:
        """
        Postprocess a batch.

        # Parameters

        batch: `Dict[str, TensorField]`
            The batch that was passed to the model's forward function.

        output_dict: `Dict`
            The output of the model's forward function on the batch

        data_loader: `DataLoader`
            The dataloader to be used.

        output_postprocess_function: `Callable`, optional (default=`None`)
            If you have a function to preprocess only the outputs (
            i.e. `model.make_human_readable`), use this parameter to have it
            called on the output dict.

        # Returns

        postprocessed: `str`
            The postprocessed batches as strings
        """
        raise NotImplementedError("__call__")

    default_implementation = "simple"


@Serializer.register("simple")
class SimpleSerializer(Serializer):
    """
    Very simple serializer. Only sanitizes the batches and outputs. Will use
     a passed serializer function for the outputs if it exists.
    """

    def _to_params(self) -> Dict[str, Any]:
        return {"type": "simple"}

    def __call__(
        self,
        batch: Dict[str, TensorField],
        output_dict: Dict,
        data_loader: DataLoader,
        output_postprocess_function: Optional[Callable] = None,
    ):
        """
        Serializer a batch.

        # Parameters

        batch: `Dict[str, TensorField]`
            The batch that was passed to the model's forward function.

        output_dict: `Dict`
            The output of the model's forward function on the batch

        data_loader: `DataLoader`
            The dataloader to be used.

        output_postprocess_function: `Callable`, optional (default=`None`)
            If you have a function to preprocess only the outputs (
            i.e. `model.make_human_readable`), use this parameter to have it
            called on the output dict.

        # Returns

        serialized: `str`
            The serialized batches as strings
        """
        if batch is None:
            raise ValueError("Serializer got a batch that is None")
        if output_dict is None:
            raise ValueError("Serializer got an output_dict that is None")

        serialized = sanitize(batch)
        if output_postprocess_function is not None:
            serialized.update(sanitize(output_postprocess_function(output_dict)))
        else:
            serialized.update(sanitize(output_dict))

        return json.dumps(serialized)
