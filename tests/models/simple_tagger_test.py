# pylint: disable=invalid-name
import numpy

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.nn.util import arrays_to_variables
from allennlp.testing.test_case import AllenNlpTestCase


class SimpleTaggerTest(AllenNlpTestCase):

    def setUp(self):
        super(SimpleTaggerTest, self).setUp()
        self.write_sequence_tagging_data()

        dataset = SequenceTaggingDatasetReader().read(self.TRAIN_FILE)
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
                        "input_size": 5,
                        "hidden_size": 7,
                        "num_layers": 2
                        }
                })

        self.model = SimpleTagger.from_params(self.vocab, params)

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.model, self.dataset)

    def test_forward_pass_runs_correctly(self):
        training_arrays = self.dataset.as_arrays()
        _ = self.model.forward(**arrays_to_variables(training_arrays))

    def test_tag_returns_distributions_per_token(self):
        text = TextField(["This", "is", "a", "sentence"], token_indexers={"tokens": SingleIdTokenIndexer()})
        output = self.model.tag(text)
        possible_tags = self.vocab.get_index_to_token_vocabulary("tags").values()
        for tag in output["tags"]:
            assert tag in possible_tags
        # Predictions are a distribution.
        numpy.testing.assert_almost_equal(numpy.sum(output["class_probabilities"], -1),
                                          numpy.array([1, 1, 1, 1]))
