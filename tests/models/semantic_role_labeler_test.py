# pylint: disable=no-self-use,invalid-name
import subprocess
import os

from flaky import flaky
import numpy

from allennlp.common.testing import ModelTestCase
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models.semantic_role_labeler import convert_bio_tags_to_conll_format
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file


class SemanticRoleLabelerTest(ModelTestCase):
    def setUp(self):
        super(SemanticRoleLabelerTest, self).setUp()
        self.set_up_model('tests/fixtures/srl/experiment.json', 'tests/fixtures/conll_2012')

    def test_srl_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_tag_returns_distributions_per_token(self):
        text = TextField(["This", "is", "a", "sentence"], token_indexers={"tokens": SingleIdTokenIndexer()})
        verb_indicator = SequenceLabelField([0, 1, 0, 0], text)

        output = self.model.tag(text, verb_indicator)
        possible_tags = self.vocab.get_index_to_token_vocabulary("labels").values()
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
