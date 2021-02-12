from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.data.data_loaders import SimpleDataLoader
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


class TestRegexOptimizer(AllenNlpTestCase):
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

    def test_multiple_optimizers(self):
        optimizer_params = Params({
                                    "type": "regex",
                                    "optimizers": [
                                        {"name": "default", "type": "adam", "lr": 1},
                                        {"name": "embedder", "type": "adam", "lr": 2},
                                        {"name": "encoder", "type": "adam", "lr": 3},
                                    ],
                                    "parameter_groups": [
                                        [["^text_field_embedder"], {"name": "embedder", "betas": (0.9, 0.98), "lr": 2, "weight_decay": 0.01}],
                                        [["^encoder.*bias"], {"name": "encoder", "lr": 0.001}],
                                        [["^encoder.*weight"], {"name": "encoder", "lr": 0.002}],
                                        [["^tag_projection_layer.*.weight$"], {"lr": 5}]
                                    ],
                                    })
        
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)
        
        for i, (name, optimizer) in enumerate(optimizer._grouped_optimizers.items()):
            if i == 0:
                assert name == "embedder", "Optimizers were not initialized in the same order as parameter groups."
                param_groups = optimizer.param_groups
                assert len(param_groups) == 1
                for param_group in param_groups:
                    assert param_group["betas"] == (0.9, 0.98)
                    assert param_group["lr"] == 2
                    assert param_group["weight_decay"] == 0.01
            elif i == 1:
                assert name == "encoder", "Optimizers were not initialized in the same order as parameter groups."
                param_groups = optimizer.param_groups
                # The optimizer can have sub-groups with different options.
                assert len(param_groups) == 2
                for i, param_group in enumerate(param_groups):
                    if i == 0:
                        assert param_group["lr"] == 0.001
                    elif i == 1:
                        assert param_group["lr"] == 0.002    
            elif i == 2:
                assert name == "default", "Optimizers were not initialized in the same order as parameter groups."
                param_groups = optimizer.param_groups
                # Default group gets assigned any group without a 'name' parameter in `parameter_groups`
                # as well as any parameters which didn't match one of the given regexes.
                assert len(param_groups) == 2

    def test_optimizer_params(self):
        optimizer_params = Params({
                                    "type": "regex",
                                    "optimizers": [
                                        {"name": "default", "type": "adam", "lr": 1},
                                        {"name": "embedder", "type": "adam", "lr": 2},
                                        {"name": "encoder", "type": "adam", "lr": 3},        
                                    ],
                                    "parameter_groups": [
                                        [["^text_field_embedder"], {"name": "embedder", "betas": (0.9, 0.98), "lr": 2, "weight_decay": 0.01}],
                                        [["^encoder.*bias"], {"name": "encoder", "lr": 0.001}],
                                        [["^encoder.*weight"], {"name": "encoder", "lr": 0.002}],
                                        [["^tag_projection_layer.*.weight$"], {"lr": 5}]
                                    ],
                                    })
        import torch
       
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)

        # When the RegexOptimizer is initialized, `optimizer.param_groups` stores the parameter groups.
        # These parameter groups are assigned to their own optimizer by the RegexOptimizer.
        # Check that changes to the parameters in optimizer.param_groups affect the parameters in optimizer._grouped_optimizers.
        regex_optimizer_params = []
        regex_optimizer_grouped_optimizer_params = []

        for param_group in optimizer.param_groups:
            params = param_group["params"]
            for param in params:
                param.data.zero_()
                regex_optimizer_params.append(id(param))

        # Check that the parameters of the sub-optimizers were also changed.
        for optimizer in optimizer._grouped_optimizers.values():
            for param_group in optimizer.param_groups:
                params = param_group["params"]
                for param in params:
                    regex_optimizer_grouped_optimizer_params.append(id(param))
                    assert param.sum() == 0, "Param has non-zero values."

        # As the optimizers are created in accordance with the order of groups in `parameter_groups`, the order of parameters should be deterministic.
        assert regex_optimizer_params == regex_optimizer_grouped_optimizer_params


class TestDenseSparseAdam(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.instances = list(
            SequenceTaggingDatasetReader().read(
                self.FIXTURES_ROOT / "data" / "sequence_tagging.tsv"
            )
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
        for instance in self.instances:
            instance.index_fields(self.vocab)
        GradientDescentTrainer(self.model, optimizer, SimpleDataLoader(self.instances, 2)).train()