import copy
import re

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


class TestMultiOptimizer(AllenNlpTestCase):
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
        optimizer_params = Params(
            {
                "type": "multi",
                "optimizers": {
                    "embedder": {"type": "adam", "lr": 2},
                    "encoder": {"type": "adam", "lr": 3},
                    "default": {"type": "adam", "lr": 1},
                },
                "parameter_groups": [
                    [
                        ["^text_field_embedder"],
                        {
                            "optimizer_name": "embedder",
                            "betas": (0.9, 0.98),
                            "lr": 2,
                            "weight_decay": 0.01,
                        },
                    ],
                    [["^encoder.*bias"], {"optimizer_name": "encoder", "lr": 0.001}],
                    [["^encoder.*weight"], {"optimizer_name": "encoder", "lr": 0.002}],
                    [["^tag_projection_layer.*.weight$"], {"lr": 5}],
                ],
            }
        )
        optimizer_params_copy = copy.deepcopy(optimizer_params)

        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)

        # make sure that every parameter is assigned to exactly one optimizer
        regex_to_optimizer_name = {
            regex: params.get("optimizer_name", "default")
            for regexes, params in optimizer_params_copy["parameter_groups"]
            for regex in regexes
        }
        for parameter_name, parameter in parameters:
            optimizer_names_for_parameter = [
                optimizer_name
                for regex, optimizer_name in regex_to_optimizer_name.items()
                if re.search(regex, parameter_name)
            ]
            if len(optimizer_names_for_parameter) <= 0:
                optimizer_names_for_parameter = ["default"]
            assert (
                len(set(optimizer_names_for_parameter)) == 1
            ), f"Parameter {parameter_name} is assigned to more than one optimizer."
            assert (
                len(optimizer_names_for_parameter) == 1
            ), f"Parameter {parameter_name} is assigned to the same optimizer more than once."

            # make sure the parameter is really assigned to that optimizer
            inner_optimizer = optimizer.optimizers[optimizer_names_for_parameter[0]]
            parameters_in_optimizer = [
                id(p) for pg in inner_optimizer.param_groups for p in pg["params"]
            ]
            assert id(parameter) in parameters_in_optimizer

        # make sure the optimizer parameters propagated properly
        for i, (name, inner_optimizer) in enumerate(optimizer.optimizers.items()):
            if i == 0:
                assert (
                    name == "embedder"
                ), "Optimizers were not initialized in the same order as parameter groups."
                param_groups = inner_optimizer.param_groups
                assert len(param_groups) == 1 + 1  # one extra for the empty default group
                assert (
                    len(param_groups[-1]["params"]) == 0
                )  # default group for an inner optimizer should be empty
                assert param_groups[0]["betas"] == (0.9, 0.98)
                assert param_groups[0]["lr"] == 2
                assert param_groups[0]["weight_decay"] == 0.01
            elif i == 1:
                assert (
                    name == "encoder"
                ), "Optimizers were not initialized in the same order as parameter groups."
                param_groups = inner_optimizer.param_groups
                # The optimizer can have sub-groups with different options.
                assert len(param_groups) == 2 + 1  # one extra for the empty default group
                assert (
                    len(param_groups[-1]["params"]) == 0
                )  # default group for an inner optimizer should be empty
                for i, param_group in enumerate(param_groups):
                    if i == 0:
                        assert param_group["lr"] == 0.001
                    elif i == 1:
                        assert param_group["lr"] == 0.002
            elif i == 2:
                assert (
                    name == "default"
                ), "Optimizers were not initialized in the same order as parameter groups."
                param_groups = inner_optimizer.param_groups
                # Default group gets assigned any group without an 'optimizer_name' parameter in `parameter_groups`
                # as well as any parameters which didn't match one of the given regexes.
                assert len(param_groups) == 2

    def test_optimizer_params(self):
        optimizer_params = Params(
            {
                "type": "multi",
                "optimizers": {
                    "default": {"type": "adam", "lr": 1},
                    "embedder": {"type": "adam", "lr": 2},
                    "encoder": {"type": "adam", "lr": 3},
                },
                "parameter_groups": [
                    [
                        ["^text_field_embedder"],
                        {
                            "optimizer_name": "embedder",
                            "betas": (0.9, 0.98),
                            "lr": 2,
                            "weight_decay": 0.01,
                        },
                    ],
                    [["^encoder.*bias"], {"optimizer_name": "encoder", "lr": 0.001}],
                    [["^encoder.*weight"], {"optimizer_name": "encoder", "lr": 0.002}],
                    [["^tag_projection_layer.*.weight$"], {"lr": 5}],
                ],
            }
        )

        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(model_parameters=parameters, params=optimizer_params)

        # When the MultiOptimizer is initialized, `optimizer.param_groups` stores the parameter groups.
        # These parameter groups are assigned to their own optimizer by the MultiOptimizer.
        # Check that changes to the parameters in optimizer.param_groups affect the parameters in
        # optimizer._grouped_optimizers.
        regex_optimizer_params = set()
        regex_optimizer_grouped_optimizer_params = set()

        for param_group in optimizer.param_groups:
            # Each param_group should have optimizer options visible so they can be used by schedulers.
            lr = param_group["lr"]
            assert lr > 0
            params = param_group["params"]
            for param in params:
                param.data.zero_()
                regex_optimizer_params.add(id(param))

        # Check that the parameters of the sub-optimizers were also changed.
        for optimizer in optimizer.optimizers.values():
            for param_group in optimizer.param_groups:
                params = param_group["params"]
                for param in params:
                    regex_optimizer_grouped_optimizer_params.add(id(param))
                    assert param.sum() == 0, "Param has non-zero values."

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
