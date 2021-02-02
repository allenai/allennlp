from pytest import raises
import torch
import torch.autograd as autograd

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import TokenCharactersIndexer
from allennlp.models.archival import load_archive
from allennlp.modules.token_embedders import EmptyEmbedder
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.common.testing.interpret_test import (
    FakeModelForTestingInterpret,
    FakePredictorForTestingInterpret
)
from allennlp.interpret.influence_interpreters import SimpleInfluence


class TestSimplInfluence(AllenNlpTestCase):
    def test_get_hessian_vector_product(self):
        A = torch.tensor([[1., 2.], [3., 4.]])
        v = torch.tensor([10, 20]).float()

        x = torch.nn.Parameter(torch.tensor([1, 2]).float(), requires_grad=True)
        x_loss = 1/2 * (x @ A @ x.T)
        hessian = SimpleInfluence.get_hessian_vector(x_loss, [x], [v])[0]
        ans = 1/2 * (A + A.T) @ v
        assert torch.equal(hessian, ans)

    def test_flatten_grad(self):
        A = torch.nn.Parameter(torch.tensor([[1., 2.], [3., 4.]]), requires_grad=True)
        B = torch.nn.Parameter(torch.tensor([[5., 6.], [7., 8.]]), requires_grad=True)
        flatten_grad = SimpleInfluence.flatten_grads([A, B])
        ans = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float()
        assert torch.equal(flatten_grad, ans)

    def test_get_inverse_hvp_lissa(self):
        pass
