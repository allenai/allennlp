# pylint: disable=no-self-use,invalid-name
import subprocess
import os
import numpy

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SrlReader
from allennlp.data.fields import TextField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import Model
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.models.semantic_role_labeler import convert_bio_tags_to_conll_format
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from allennlp.common.testing import AllenNlpTestCase


class SemanticRoleLabelerTest(AllenNlpTestCase):
    def setUp(self):
        super(SemanticRoleLabelerTest, self).setUp()

        dataset = SrlReader().read('tests/fixtures/conll_2012/')
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

    def test_srl_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.model, self.dataset)

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

    def test_bio_tags_correctly_convert_to_conll_format(self):
        bio_tags = ["B-ARG-1", "I-ARG-1", "O", "B-V", "B-ARGM-ADJ", "O"]
        conll_tags = convert_bio_tags_to_conll_format(bio_tags)
        assert conll_tags == ["(ARG-1*", "*)", "*", "(V*)", "(ARGM-ADJ*)", "*"]

    def test_perl_eval_script_can_run_on_printed_conll_files(self):
        bio_tags = ["B-ARG-1", "I-ARG-1", "O", "B-V", "B-ARGM-ADJ", "O"]
        sentence = ["Mark", "and", "Matt", "were", "running", "fast", "."]

        gold_file_path = os.path.join(self.TEST_DIR, "gold_conll_eval.txt")
        prediction_file_path = os.path.join(self.TEST_DIR, "prediction_conll_eval.txt")
        with open(gold_file_path, "a+") as gold_file, open(prediction_file_path, "a+") as prediction_file:
            # Use the same bio tags as prediction vs gold to make it obvious by looking
            # at the perl script output if something is wrong. Write them twice to
            # ensure that the perl script deals with multiple sentences.
            write_to_conll_eval_file(gold_file, prediction_file, 4, sentence, bio_tags, bio_tags)
            write_to_conll_eval_file(gold_file, prediction_file, 4, sentence, bio_tags, bio_tags)

        perl_script_command = ["perl", "./scripts/srl-eval.pl", prediction_file_path, gold_file_path]
        exit_code = subprocess.check_call(perl_script_command)
        assert exit_code == 0

    def test_from_file(self):
        params = Params.from_file('tests/fixtures/srl/experiment.json')
        model = Model.from_files(params)

        assert isinstance(model, SemanticRoleLabeler)
