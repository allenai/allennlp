import torch
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TensorField
from allennlp.data import Instance
from allennlp.data.data_loaders import SimpleDataLoader


from allennlp.common.testing.interpret_test import (
    DummyBilinearModelForTestingIF,
)
from allennlp.interpret.influence_interpreters import SimpleInfluence


class TestSimplInfluence(AllenNlpTestCase):
    def test_get_hessian_vector_product(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        v = torch.tensor([10, 20]).float()

        x = torch.nn.Parameter(torch.tensor([1, 2]).float(), requires_grad=True)
        x_loss = 1 / 2 * (x @ A @ x.T)
        hessian_vector_product = SimpleInfluence.get_hessian_vector_product(x_loss, [x], [v])[0]
        ans = 1 / 2 * (A + A.T) @ v
        assert torch.equal(hessian_vector_product, ans)

    def test_flatten_tensors(self):
        A = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        B = torch.nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
        flatten_grad = SimpleInfluence.flatten_tensors([A, B])
        ans = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float()
        assert torch.equal(flatten_grad, ans)

    def test_get_inverse_hvp_lissa(self):
        vs = [torch.tensor([1.0, 1.0])]
        # create a fake model
        vocab = Vocabulary()
        params = torch.tensor([1, 2]).float()
        model = DummyBilinearModelForTestingIF(vocab, params)
        used_params = list(model.parameters())

        # create a fake instance: just a matrix
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        fake_instance = Instance({"tensors": TensorField(A)})

        # wrap fake instance into dataloader
        lissa_dataloader = SimpleDataLoader([fake_instance], batch_size=1)

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
        # I tried to increase recursion depth to actually approx the inverse Hessian vector product,
        # but I suspect due to extremely small number of data point, the algorithm doesn't work well
        # on this toy example
        ans = torch.tensor([-1.5, -4.5])
        assert torch.equal(inverse_hvp, ans)

    def test_freeze_params(self):
        # create a fake model
        vocab = Vocabulary()
        params = torch.tensor([1, 2]).float()
        model = DummyBilinearModelForTestingIF(vocab, params)

        # create a fake instance: just a matrix
        assert ["x"] == [n for n, p in model.named_parameters() if p.requires_grad]
        SimpleInfluence.freeze_model(model, ["x"])
        assert [] == [n for n, p in model.named_parameters() if p.requires_grad]
