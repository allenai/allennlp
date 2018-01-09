# pylint: disable=no-self-use,invalid-name,protected-access
import os
import subprocess

import torch
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.training.metrics import SpanBasedF1Measure, Metric
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from allennlp.common.params import Params


class SpanBasedF1Test(AllenNlpTestCase):

    def setUp(self):
        super(SpanBasedF1Test, self).setUp()
        vocab = Vocabulary()
        vocab.add_token_to_namespace("O", "tags")
        vocab.add_token_to_namespace("B-ARG1", "tags")
        vocab.add_token_to_namespace("I-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARG2", "tags")
        vocab.add_token_to_namespace("I-ARG2", "tags")
        vocab.add_token_to_namespace("B-V", "tags")
        vocab.add_token_to_namespace("I-V", "tags")
        vocab.add_token_to_namespace("U-ARG1", "tags")
        vocab.add_token_to_namespace("U-ARG2", "tags")
        vocab.add_token_to_namespace("B-C-ARG1", "tags")
        vocab.add_token_to_namespace("I-C-ARG1", "tags")
        vocab.add_token_to_namespace("B-ARGM-ADJ", "tags")
        vocab.add_token_to_namespace("I-ARGM-ADJ", "tags")

        self.vocab = vocab

    def test_span_metrics_are_computed_correcly_with_prediction_map(self):
        # In this example, datapoint1 only has access to ARG1 and V labels,
        # whereas datapoint2 only has access to ARG2 and V labels.

        # gold_labels = [["O", "B-ARG1", "I-ARG1", "O", "B-V", "O"],
        #               ["B-ARG2", "I-ARG2", "O", "B-V", "I-V", "O"]]
        gold_indices = [[0, 1, 2, 0, 3, 0],
                        [1, 2, 0, 3, 4, 0]]
        prediction_map_indices = [[0, 1, 2, 5, 6],
                                  [0, 3, 4, 5, 6]]

        gold_tensor = torch.Tensor(gold_indices)
        prediction_map_tensor = torch.Tensor(prediction_map_indices)

        prediction_tensor = torch.rand([2, 6, 5])
        prediction_tensor[0, 0, 0] = 1
        prediction_tensor[0, 1, 1] = 1 # (True Positive - ARG1
        prediction_tensor[0, 2, 2] = 1 # *)
        prediction_tensor[0, 3, 0] = 1
        prediction_tensor[0, 4, 3] = 1 # (True Positive - V)
        prediction_tensor[0, 5, 1] = 1 # (False Positive - ARG1)
        prediction_tensor[1, 0, 0] = 1 # (False Negative - ARG2
        prediction_tensor[1, 1, 0] = 1 # *)
        prediction_tensor[1, 2, 0] = 1
        prediction_tensor[1, 3, 3] = 1 # (True Positive - V
        prediction_tensor[1, 4, 4] = 1 # *)
        prediction_tensor[1, 5, 1] = 1 # (False Positive - ARG2)

        metric = SpanBasedF1Measure(self.vocab, "tags")
        metric(prediction_tensor, gold_tensor, prediction_map=prediction_map_tensor)

        assert metric._true_positives["ARG1"] == 1
        assert metric._true_positives["ARG2"] == 0
        assert metric._true_positives["V"] == 2
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["ARG1"] == 0
        assert metric._false_negatives["ARG2"] == 1
        assert metric._false_negatives["V"] == 0
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["ARG1"] == 1
        assert metric._false_positives["ARG2"] == 1
        assert metric._false_positives["V"] == 0
        assert "O" not in metric._false_positives.keys()

        # Check things are accumulating correctly.
        metric(prediction_tensor, gold_tensor, prediction_map=prediction_map_tensor)
        assert metric._true_positives["ARG1"] == 2
        assert metric._true_positives["ARG2"] == 0
        assert metric._true_positives["V"] == 4
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["ARG1"] == 0
        assert metric._false_negatives["ARG2"] == 2
        assert metric._false_negatives["V"] == 0
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["ARG1"] == 2
        assert metric._false_positives["ARG2"] == 2
        assert metric._false_positives["V"] == 0
        assert "O" not in metric._false_positives.keys()

        metric_dict = metric.get_metric()

        numpy.testing.assert_almost_equal(metric_dict["recall-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-ARG1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARG1"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARG1"], 0.666666666)
        numpy.testing.assert_almost_equal(metric_dict["recall-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.75)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 0.6)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.666666666)

    def test_span_metrics_are_computed_correctly(self):
        gold_labels = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "O", "O", "O"]
        gold_indices = [self.vocab.get_token_index(x, "tags") for x in gold_labels]

        gold_tensor = torch.Tensor([gold_indices])

        prediction_tensor = torch.rand([2, 9, self.vocab.get_vocab_size("tags")])

        # Test that the span measure ignores completely masked sequences by
        # passing a mask with a fully masked row.
        mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        prediction_tensor[:, 0, 0] = 1
        prediction_tensor[:, 1, 1] = 1  # (True positive - ARG1
        prediction_tensor[:, 2, 2] = 1  # *)
        prediction_tensor[:, 3, 0] = 1
        prediction_tensor[:, 4, 0] = 1  # (False Negative - ARG2
        prediction_tensor[:, 5, 0] = 1  # *)
        prediction_tensor[:, 6, 0] = 1
        prediction_tensor[:, 7, 1] = 1  # (False Positive - ARG1
        prediction_tensor[:, 8, 2] = 1  # *)

        metric = SpanBasedF1Measure(self.vocab, "tags")
        metric(prediction_tensor, gold_tensor, mask)

        assert metric._true_positives["ARG1"] == 1
        assert metric._true_positives["ARG2"] == 0
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["ARG1"] == 0
        assert metric._false_negatives["ARG2"] == 1
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["ARG1"] == 1
        assert metric._false_positives["ARG2"] == 0
        assert "O" not in metric._false_positives.keys()

        # Check things are accumulating correctly.
        metric(prediction_tensor, gold_tensor, mask)
        assert metric._true_positives["ARG1"] == 2
        assert metric._true_positives["ARG2"] == 0
        assert "O" not in metric._true_positives.keys()
        assert metric._false_negatives["ARG1"] == 0
        assert metric._false_negatives["ARG2"] == 2
        assert "O" not in metric._false_negatives.keys()
        assert metric._false_positives["ARG1"] == 2
        assert metric._false_positives["ARG2"] == 0
        assert "O" not in metric._false_positives.keys()

        metric_dict = metric.get_metric()

        numpy.testing.assert_almost_equal(metric_dict["recall-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARG2"], 0.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-ARG1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARG1"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARG1"], 0.666666666)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 0.5)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 0.5)

    def test_span_f1_can_build_from_params(self):
        params = Params({"type": "span_f1", "tag_namespace": "tags", "ignore_classes": ["V"]})
        metric = Metric.from_params(params, self.vocab)
        assert metric._ignore_classes == ["V"]
        assert metric._label_vocabulary == self.vocab.get_index_to_token_vocabulary("tags")

    def test_span_f1_matches_perl_script_for_continued_arguments(self):
        bio_tags = ["B-ARG1", "O", "B-C-ARG1", "B-V", "B-ARGM-ADJ", "O"]
        sentence = ["Mark", "and", "Matt", "were", "running", "fast", "."]

        gold_indices = [self.vocab.get_token_index(x, "tags") for x in bio_tags]
        gold_tensor = torch.Tensor([gold_indices])
        prediction_tensor = torch.rand([1, 6, self.vocab.get_vocab_size("tags")])
        mask = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

        # Make prediction so that it is exactly correct.
        for i, tag_index in enumerate(gold_indices):
            prediction_tensor[0, i, tag_index] = 1

        metric = SpanBasedF1Measure(self.vocab, "tags")
        metric(prediction_tensor, gold_tensor, mask)
        metric_dict = metric.get_metric()

        # We merged the continued ARG1 label into a single span, so there should
        # be exactly 1 true positive for ARG1 and nothing present for C-ARG1
        assert metric._true_positives["ARG1"] == 1
        # The labels containing continuation references get merged into
        # the labels that they continue, so they should never appear in
        # the precision/recall counts.
        assert "C-ARG1" not in metric._true_positives.keys()
        assert metric._true_positives["V"] == 1
        assert metric._true_positives["ARGM-ADJ"] == 1

        numpy.testing.assert_almost_equal(metric_dict["recall-ARG1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARG1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARG1"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-V"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-ARGM-ADJ"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-ARGM-ADJ"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-ARGM-ADJ"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["recall-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["precision-overall"], 1.0)
        numpy.testing.assert_almost_equal(metric_dict["f1-measure-overall"], 1.0)

        # Check that the number of true positive ARG1 labels is the same as the perl script's output:
        gold_file_path = os.path.join(self.TEST_DIR, "gold_conll_eval.txt")
        prediction_file_path = os.path.join(self.TEST_DIR, "prediction_conll_eval.txt")
        with open(gold_file_path, "a+") as gold_file, open(prediction_file_path, "a+") as prediction_file:
            # Use the same bio tags as prediction vs gold to make it obvious by looking
            # at the perl script output if something is wrong.
            write_to_conll_eval_file(gold_file, prediction_file, 4, sentence, bio_tags, bio_tags)
        # Run the official perl script and collect stdout.
        perl_script_command = ["perl", "./scripts/srl-eval.pl", prediction_file_path, gold_file_path]
        stdout = subprocess.check_output(perl_script_command, universal_newlines=True)
        stdout_lines = stdout.split("\n")
        # Parse the stdout of the perl script to find the ARG1 row (this happens to be line 8).
        num_correct_arg1_instances_from_perl_evaluation = int([token for token in
                                                               stdout_lines[8].split(" ") if token][1])
        assert num_correct_arg1_instances_from_perl_evaluation == metric._true_positives["ARG1"]
