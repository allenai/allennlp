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
from allennlp.data.fields import TensorField
from allennlp.data import Instance
from allennlp.data.data_loaders import SimpleDataLoader


from allennlp.common.testing.interpret_test import (
    FakeModelForTestingInterpret,
    FakePredictorForTestingInterpret,
    DummyBilinearModelForTestingIF,
)
from allennlp.interpret.influence_interpreters import SimpleInfluence


class TestSimplInfluence(AllenNlpTestCase):
    def test_get_hessian_vector_product(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        v = torch.tensor([10, 20]).float()

        x = torch.nn.Parameter(torch.tensor([1, 2]).float(), requires_grad=True)
        x_loss = 1 / 2 * (x @ A @ x.T)
        hessian = SimpleInfluence.get_hessian_vector_product(x_loss, [x], [v])[0]
        ans = 1 / 2 * (A + A.T) @ v
        assert torch.equal(hessian, ans)

    def test_flatten_tensors(self):
        A = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        B = torch.nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        flatten_grad = SimpleInfluence.flatten_tensors([A, B])
        ans = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float()
        assert torch.equal(flatten_grad, ans)

    def test_get_inverse_hvp_lissa(self):
        vs = [torch.tensor([1.0, 1.0])]
        # create a fake instance: just a matrix
        # params = torch.tensor([1, 2, 3]).float()
        params = torch.tensor([1, 2]).float()
        # A = torch.tensor([[1., 2., 3.], [4., 5, 6.], [7, 8, 9]])
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fake_instance = Instance({"tensors": TensorField(A)})
        lissa_dataloader = SimpleDataLoader([fake_instance], batch_size=1)
        vocab = Vocabulary()
        model = DummyBilinearModelForTestingIF(vocab, params)
        used_params = list(model.parameters())
        inverse_hvp = SimpleInfluence.get_inverse_hvp_lissa(
            vs=vs,
            model=model,
            used_params=used_params,
            lissa_dataloader=lissa_dataloader,
            num_samples=1,
            recursion_depth=1,
            damping=0,
            scale=1,
        )
        ans = torch.tensor([-1.5, -4.5])  # make this general for recursion_depth > 1
        assert torch.equal(inverse_hvp, ans)
