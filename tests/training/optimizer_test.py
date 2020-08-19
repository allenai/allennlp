from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.data import PyTorchDataLoader
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import Optimizer


class TestOptimizer(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.instances = SequenceTaggingDatasetReader().read(
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
        )
        vocab = Vocabulary.from_instances(self.instances)
        self.model_params = Params(
            {
                "text_field_embedder": {
                    "token_embedders": {"tokens": {"type": "embedding", "embedding_dim": 5}}
                },
                "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
            }
        )
        self.model = SimpleTagger.from_params(vocab=vocab, params=self.model_params)

    def test_optimizer_basic(self):
        optimizer_params = Params({"type": "sgd", "lr": 1})
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)
        param_groups = optimizer.param_groups
        assert len(param_groups) == 1
        assert param_groups[0]["lr"] == 1

    def test_optimizer_parameter_groups(self):
        optimizer_params = Params(
            {
                "type": "sgd",
                "lr": 1,
                "momentum": 5,
                "parameter_groups": [
                    # the repeated "bias_" checks a corner case
                    # NOT_A_VARIABLE_NAME displays a warning but does not raise an exception
                    [["weight_i", "bias_", "bias_", "NOT_A_VARIABLE_NAME"], {"lr": 2}],
                    [["tag_projection_layer"], {"lr": 3}],
                    [["^text_field_embedder.*$"], {"requires_grad": False}],
                ],
            }
        )

        # Before initializing the optimizer all params in this module will still require grad.
        assert all([param.requires_grad for param in self.model.text_field_embedder.parameters()])

        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)
        param_groups = optimizer.param_groups

        # After initializing the optimizer, requires_grad should be false for all params in this module.
        assert not any(
            [param.requires_grad for param in self.model.text_field_embedder.parameters()]
        )

        assert len(param_groups) == 3
        assert param_groups[0]["lr"] == 2
        assert param_groups[1]["lr"] == 3
        # base case uses default lr
        assert param_groups[2]["lr"] == 1
        for k in range(3):
            assert param_groups[k]["momentum"] == 5

        # all LSTM parameters except recurrent connections (those with weight_h in name)
        assert len(param_groups[0]["params"]) == 6
        # just the projection weight and bias
        assert len(param_groups[1]["params"]) == 2
        # the recurrent connections left in the default group
        assert len(param_groups[2]["params"]) == 2


class TestDenseSparseAdam(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.instances = SequenceTaggingDatasetReader().read(
            self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
        )
        self.vocab = Vocabulary.from_instances(self.instances)
        self.model_params = Params(
            {
                "text_field_embedder": {
                    "token_embedders": {
                        "tokens": {"type": "embedding", "embedding_dim": 5, "sparse": True}
                    }
                },
                "encoder": {"type": "lstm", "input_size": 5, "hidden_size": 7, "num_layers": 2},
            }
        )
        self.model = SimpleTagger.from_params(vocab=self.vocab, params=self.model_params)

    def test_can_optimise_model_with_dense_and_sparse_params(self):
        optimizer_params = Params({"type": "dense_sparse_adam"})
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)
        self.instances.index_with(self.vocab)
        GradientDescentTrainer(self.model, optimizer, PyTorchDataLoader(self.instances, 2)).train()
