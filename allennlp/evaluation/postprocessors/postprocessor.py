from collections import defaultdict
from typing import Union, List, Dict, Any, Tuple, Optional
from os import PathLike
from pathlib import Path
import torch
from itertools import groupby
import logging
import json
import numpy as np

from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, sanitize, int_to_device, END_SYMBOL, START_SYMBOL
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
            data_loader: DataLoader
    ) -> str:
        raise NotImplementedError("__call__")

    default_implementation = "simple"


@Postprocessor.register("simple")
class SimplePostprocessor(Postprocessor):
    def _to_params(self) -> Dict[str, Any]:
        return {
            "type": "simple"
        }

    def __call__(
            self,
            batch: Dict[str, TensorField],
            output_dict: Dict,
            data_loader: DataLoader
    ):
        if batch is None:
            raise ValueError("Postprocessor got a batch that is None")
        if output_dict is None:
            raise ValueError("Postprocessor got an output_dict that is None")
        return json.dumps(sanitize({**batch, **output_dict}))
