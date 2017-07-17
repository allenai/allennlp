from typing import Dict

import torch
import torch.nn
from torch.autograd import Variable

from allennlp.service.models.types import Model, JSON

N, D_in, H, D_out = 640, 10000, 100, 10  # pylint: disable=invalid-name

def create_model() -> torch.nn.Module:
    """
    not a very interesting model, just does a large matrix multiplication
    see https://github.com/jcjohnson/pytorch-examples#pytorch-nn
    """

    model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                torch.nn.ReLU(),
                                torch.nn.Linear(H, D_out))
    return model

def shared_pytorch_model() -> Model:
    # single underlying model shared by every call to `run`
    model = create_model()

    def run(blob: JSON) -> JSON:
        # create a random matrix that depends on the request
        i = int(blob["input"])
        input_var = Variable(torch.randn(N, D_in) / (i + 1))

        # make a prediction and convert it to a list
        y_pred = model(input_var)
        output = y_pred.data.tolist()

        return {"input": i, "model_name": "pytorch", "output": output}

    return run

def models() -> Dict[str, Model]:
    return {'pytorch': shared_pytorch_model()}
