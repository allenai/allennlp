# pylint: disable=no-self-use,invalid-name,protected-access
import torch
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from allennlp.training.metrics import F1Measure, SpanBasedF1Measure, Metric
from allennlp.common.params import Params

class CategoricalAccuracyTest(AllenNlpTestCase):
    def test_categorical_accuracy(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

    def test_top_k_categorical_accuracy(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 1.0

    def test_top_k_categorical_accuracy_accumulates_and_resets_correctly(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 3])
        accuracy(predictions, targets)
        accuracy(predictions, targets)
        accuracy(predictions, torch.Tensor([4, 4]))
        accuracy(predictions, torch.Tensor([4, 4]))
        actual_accuracy = accuracy.get_metric(reset=True)
        assert actual_accuracy == 0.50
        assert accuracy.correct_count == 0.0
        assert accuracy.total_count == 0.0

    def test_top_k_categorical_accuracy_respects_mask(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.5, 0.2, 0.0]])
        targets = torch.Tensor([0, 3, 0])
        mask = torch.Tensor([0, 1, 1])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric()
        assert actual_accuracy == 0.50

    def test_top_k_categorical_accuracy_works_for_sequences(self):
        accuracy = CategoricalAccuracy(top_k=2)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]]])
        targets = torch.Tensor([[0, 3, 4],
                                [0, 1, 4]])
        accuracy(predictions, targets)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.6666666)

        # Test the same thing but with a mask:
        mask = torch.Tensor([[0, 1, 1],
                             [1, 0, 1]])
        accuracy(predictions, targets, mask)
        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0.50)

    def test_top_k_categorical_accuracy_catches_exceptions(self):
        accuracy = CategoricalAccuracy()
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            accuracy(predictions, out_of_range_labels)

class BooleanAccuracyTest(AllenNlpTestCase):
    def test_accuracy_computation(self):
        accuracy = BooleanAccuracy()
        predictions = torch.Tensor([[0, 1],
                                    [2, 3],
                                    [4, 5],
                                    [6, 7]])
        targets = torch.Tensor([[0, 1],
                                [2, 2],
                                [4, 5],
                                [7, 7]])
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 2. / 4

        mask = torch.ones(4, 2)
        mask[1, 1] = 0
        accuracy(predictions, targets, mask)
        assert accuracy.get_metric() == 5. / 8

        targets[1, 1] = 3
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 8. / 12

        accuracy.reset()
        accuracy(predictions, targets)
        assert accuracy.get_metric() == 3. / 4

class F1MeasureTest(AllenNlpTestCase):
    def test__f1_measure_catches_exceptions(self):
        f1_measure = F1Measure(0)
        predictions = torch.rand([5, 7])
        out_of_range_labels = torch.Tensor([10, 3, 4, 0, 1])
        with pytest.raises(ConfigurationError):
            f1_measure(predictions, out_of_range_labels)

    def test_f1_measure(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 3.
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 2.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)

        # Test the same thing with a mask:
        mask = torch.Tensor([1, 0, 1, 1, 1, 0])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.5)
        numpy.testing.assert_almost_equal(f1, 0.6666666666)

    def test_f1_measure_accumulates_and_resets_correctly(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        # [True Positive, True Negative, True Negative,
        #  False Negative, True Negative, False Negative]
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        f1_measure(predictions, targets)
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 6.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 4.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.333333333)
        numpy.testing.assert_almost_equal(f1, 0.499999999)
        assert f1_measure._true_positives == 0.0
        assert f1_measure._true_negatives == 0.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 0.0

    def test_f1_measure_works_for_sequences(self):
        f1_measure = F1Measure(positive_label=0)
        predictions = torch.Tensor([[[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]],
                                    [[0.35, 0.25, 0.1, 0.1, 0.2],
                                     [0.1, 0.6, 0.1, 0.2, 0.0],
                                     [0.1, 0.6, 0.1, 0.2, 0.0]]])
        # [[True Positive, True Negative, True Negative],
        #  [True Positive, True Negative, False Negative]]
        targets = torch.Tensor([[0, 3, 4],
                                [0, 1, 0]])
        f1_measure(predictions, targets)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 2.0
        assert f1_measure._true_negatives == 3.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        f1_measure.reset()
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.666666666)
        numpy.testing.assert_almost_equal(f1, 0.8)

        # Test the same thing with a mask:
        mask = torch.Tensor([[0, 1, 0],
                             [1, 1, 1]])
        f1_measure(predictions, targets, mask)
        precision, recall, f1 = f1_measure.get_metric()
        assert f1_measure._true_positives == 1.0
        assert f1_measure._true_negatives == 2.0
        assert f1_measure._false_positives == 0.0
        assert f1_measure._false_negatives == 1.0
        numpy.testing.assert_almost_equal(precision, 1.0)
        numpy.testing.assert_almost_equal(recall, 0.5)
        numpy.testing.assert_almost_equal(f1, 0.66666666666)


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
        self.vocab = vocab

    def test_span_based_f1_extracts_correct_spans(self):
        metric = SpanBasedF1Measure(self.vocab, tag_namespace="tags")
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        indices = [self.vocab.get_token_index(x, "tags") for x in tag_sequence]
        spans = metric._extract_spans(indices)
        assert spans == {((1, 2), "ARG1"), ((4, 5), "ARG2"), ((6, 6), "ARG1"), ((7, 7), "ARG2")}

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = ["O", "B-ARG1", "I-ARG1", "O", "I-ARG1", "B-ARG2", "I-ARG2", "B-ARG1", "I-ARG2", "I-ARG2"]
        indices = [self.vocab.get_token_index(x, "tags") for x in tag_sequence]
        spans = metric._extract_spans(indices)
        assert spans == {((1, 2), "ARG1"), ((5, 6), "ARG2"), ((7, 7), "ARG1"),
                         ((4, 4), "ARG1"), ((8, 9), "ARG2")}

    def test_span_based_f1_ignores_specified_tags(self):
        metric = SpanBasedF1Measure(self.vocab, "tags", ["ARG1", "V"])

        tag_sequence = ["B-V", "I-V", "O", "B-ARG1", "I-ARG1",
                        "O", "B-ARG2", "I-ARG2", "B-ARG1", "B-ARG2"]
        indices = [self.vocab.get_token_index(x, "tags") for x in tag_sequence]
        spans = metric._extract_spans(indices)
        assert spans == {((6, 7), "ARG2"), ((9, 9), "ARG2")}

    def test_span_metrics_are_computed_correctly(self):
        gold_labels = ["O", "B-ARG1", "I-ARG1", "O", "B-ARG2", "I-ARG2", "O", "O", "O"]
        gold_indices = [self.vocab.get_token_index(x, "tags") for x in gold_labels]

        gold_tensor = torch.Tensor([gold_indices])

        prediction_tensor = torch.rand([1, 9, self.vocab.get_vocab_size("tags")])
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
        metric(prediction_tensor, gold_tensor)

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
        metric(prediction_tensor, gold_tensor)
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
