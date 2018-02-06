# pylint: disable=invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.training.optimizers import Optimizer


class TestOptimizer(AllenNlpTestCase):
    def setUp(self):
        super(TestOptimizer, self).setUp()
        self.instances = SequenceTaggingDatasetReader().read('tests/fixtures/data/sequence_tagging.tsv')
        vocab = Vocabulary.from_instances(self.instances)
        self.model_params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "stacked_encoder": {
                        "type": "lstm",
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })
        self.model = SimpleTagger.from_params(vocab, self.model_params)

    def test_optimizer_basic(self):
        optimizer_params = Params({
                "type": "sgd",
                "lr": 1
        })
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, optimizer_params)
        param_groups = optimizer.param_groups
        assert len(param_groups) == 1
        assert param_groups[0]['lr'] == 1

    def test_optimizer_parameter_groups(self):
        optimizer_params = Params({
                "type": "sgd",
                "lr": 1,
                "momentum": 5,
                "parameter_groups": [
                        # the repeated "bias_" checks a corner case
                        # NOT_A_VARIABLE_NAME displays a warning but does not raise an exception
                        [["weight_i", "bias_", "bias_", "NOT_A_VARIABLE_NAME"], {'lr': 2}],
                        [["tag_projection_layer"], {'lr': 3}],
                ]
        })
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, optimizer_params)
        param_groups = optimizer.param_groups

        assert len(param_groups) == 3
        assert param_groups[0]['lr'] == 2
        assert param_groups[1]['lr'] == 3
        # base case uses default lr
        assert param_groups[2]['lr'] == 1
        for k in range(3):
            assert param_groups[k]['momentum'] == 5

        # all LSTM parameters except recurrent connections (those with weight_h in name)
        assert len(param_groups[0]['params']) == 6
        # just the projection weight and bias
        assert len(param_groups[1]['params']) == 2
        # the embedding + recurrent connections left in the default group
        assert len(param_groups[2]['params']) == 3
