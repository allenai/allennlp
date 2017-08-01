import numpy

from allennlp.common import Params
from allennlp.common.tensor import arrays_to_variables
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SrlReader
from allennlp.data.fields import TextField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.testing.test_case import AllenNlpTestCase


class SemanticRoleLabelerTest(AllenNlpTestCase):

    def setUp(self):
        super(SemanticRoleLabelerTest, self).setUp()
        self.write_conll_2012_data()

        dataset = SrlReader().read(self.CONLL_TRAIN_DIR)
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset

        params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "stacked_encoder": {
                        "type": "lstm",
                        "input_size": 6,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })

        self.model = SemanticRoleLabeler.from_params(self.vocab, params)

    def test_srl_tagger_saves_and_loads(self):
        self.ensure_model_saves_and_loads(self.model, self.dataset)

    def test_forward_pass_runs_correctly(self):
        training_arrays = self.dataset.as_arrays()
        model_inputs = arrays_to_variables(training_arrays)
        _ = self.model.forward(**model_inputs)

    def test_tag_returns_distributions_per_token(self):
        text = TextField(["This", "is", "a", "sentence"], token_indexers={"tokens": SingleIdTokenIndexer()})
        verb_indicator = IndexField(1, text)

        output = self.model.tag(text, verb_indicator)
        possible_tags = self.vocab.get_index_to_token_vocabulary("tags").values()
        for tag in output["tags"]:
            assert tag in possible_tags
        # Predictions are a distribution.
        numpy.testing.assert_almost_equal(numpy.sum(output["class_probabilities"], -1),
                                          numpy.array([1, 1, 1, 1]))
