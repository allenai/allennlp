import torch
import torch.nn
from torch.autograd import Variable

from allennlp.service.servable import Servable, JSONDict

class MatrixMultiplier(Servable):
    def __init__(self, N: int = 640, D_in: int = 10000, H: int = 100, D_out: int = 10) -> None:
        """
        not a very interesting model, just does a large matrix multiplication
        see https://github.com/jcjohnson/pytorch-examples#pytorch-nn
        """
        self.N = N          # pylint: disable=invalid-name
        self.D_in = D_in    # pylint: disable=invalid-name
        self.model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(H, D_out))

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        # create a random matrix that depends on the request
        i = int(inputs["input"])
        input_var = Variable(torch.randn(self.N, self.D_in) / (i + 1))

        # make a prediction and convert it to a list
        y_pred = self.model(input_var)
        output = y_pred.data.tolist()

        return {"input": i, "model_name": "matrix_multiplier", "output": output}
