from collections import defaultdict
from typing import Optional, Dict, Any, Callable
import logging
import json

from allennlp.common.util import sanitize
from allennlp.data.fields import TensorField
from allennlp.nn import util as nn_util
from allennlp.common import Registrable, Params
from allennlp.models import Model
from allennlp.data import DataLoader, Vocabulary

logger = logging.getLogger(__name__)


class Postprocessor(Registrable):
    """
    General Postprocessor class for turning batches into human readable data
    """

    def __call__(
            self,
            batch: Dict[str, TensorField],
            output_dict: Dict,
            data_loader: DataLoader,
            output_postprocess_function: Optional[Callable] = None
    ) -> str:
        raise NotImplementedError("__call__")

    default_implementation = "simple"


@Postprocessor.register("simple")
class SimplePostprocessor(Postprocessor):
    """
    Very simple postprocesser. Only sanitizes the batches and outputs. Will use
     a passed postprocess function for the outputs if it exists.
    """

    def _to_params(self) -> Dict[str, Any]:
        return {
            "type": "simple"
        }

    def __call__(
            self,
            batch: Dict[str, TensorField],
            output_dict: Dict,
            data_loader: DataLoader,
            output_postprocess_function: Optional[Callable] = None
    ):
        if batch is None:
            raise ValueError("Postprocessor got a batch that is None")
        if output_dict is None:
            raise ValueError("Postprocessor got an output_dict that is None")

        postprocessed = sanitize(batch)
        if output_postprocess_function is not None:
            postprocessed.update(sanitize(output_postprocess_function(output_dict)))
        else:
            postprocessed.update(sanitize(output_dict))

        return json.dumps(postprocessed)
