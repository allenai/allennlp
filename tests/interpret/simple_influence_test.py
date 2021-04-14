import torch
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TensorField
from allennlp.data import Instance
from allennlp.models.model import Model
from allennlp.data.data_loaders import SimpleDataLoader

from allennlp.interpret import InfluenceInterpreter
from allennlp.interpret.influence_interpreters.simple_influence import (
    flatten_tensors,
    get_hessian_vector_product,
    get_inverse_hvp_lissa,
)


class DummyBilinearModelForTestingIF(Model):
    def __init__(self, vocab, params):
        super().__init__(vocab)
        self.x = torch.nn.Parameter(params.float(), requires_grad=True)

    def forward(self, tensors):
        A = tensors  # (batch_size, ..., ...)
        output_dict = {"loss": 1 / 2 * (A @ self.x @ self.x)}
        return output_dict


def test_get_hessian_vector_product():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([10, 20]).float()

    x = torch.nn.Parameter(torch.tensor([1, 2]).float(), requires_grad=True)
    x_loss = 1 / 2 * (x @ A @ x.T)
    hessian_vector_product = get_hessian_vector_product(x_loss, [x], [v])[0]
    ans = 1 / 2 * (A + A.T) @ v
    assert torch.equal(hessian_vector_product, ans)


def test_flatten_tensors():
    A = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    B = torch.nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    flatten_grad = flatten_tensors([A, B])
    ans = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float()
    assert torch.equal(flatten_grad, ans)


def test_get_inverse_hvp_lissa():
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
    lissa_data_loader = SimpleDataLoader([fake_instance], batch_size=1)

    inverse_hvp = get_inverse_hvp_lissa(
        vs=vs,
        model=model,
        used_params=used_params,
        lissa_data_loader=lissa_data_loader,
        damping=0.0,
        num_samples=1,
        recursion_depth=1,
        scale=1.0,
    )
    # I tried to increase recursion depth to actually approx the inverse Hessian vector product,
    # but I suspect due to extremely small number of data point, the algorithm doesn't work well
    # on this toy example
    ans = torch.tensor([-1.5, -4.5])
    assert torch.equal(inverse_hvp, ans)


class TestSimpleInfluence(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.archive_path = (
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        self.data_path = (
            self.FIXTURES_ROOT / "data" / "text_classification_json" / "imdb_corpus.jsonl"
        )

    def test_simple_influence(self):
        # NOTE: We use the same data here for test and train, which is pointless in
        # real life but convenient here.
        si = InfluenceInterpreter.from_path(
            self.archive_path, train_data_path=self.data_path, recursion_depth=3
        )
        results = si.interpret_from_file(self.data_path, k=1)
        assert len(results) == 3
        for result in results:
            assert len(result.top_k) == len(result.bottom_k) == 1
