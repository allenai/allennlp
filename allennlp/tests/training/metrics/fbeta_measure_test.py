# pylint: disable=no-self-use,invalid-name,protected-access
import numpy
import torch
from allennlp.common.checks import ConfigurationError

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import FBetaMeasure


class FBetaMeasureTest(AllenNlpTestCase):
    def test_config_errors(self):
        # Bad beta
        self.assertRaises(ConfigurationError, FBetaMeasure,
                          beta=0.0)

        # Bad average option
        self.assertRaises(ConfigurationError, FBetaMeasure,
                          average='mega')

    def test_fbeta_multiclass(self):
        # Test FBetaMeasure for multiclass classification task.
        # [0, 1, 1, 1, 3, 1]
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        fbeta = FBetaMeasure()
        fbeta(predictions, targets)
        precisions, recalls, fs = fbeta.get_metric()

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 4, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [3, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._tp_sum.tolist(), [1, 1, 0, 1, 0])

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00]
        desired_fs = [(2*p*r)/(p+r) if p+r != 0.0 else 0.0
                      for p, r in zip(desired_precisions, desired_recalls)]
        numpy.testing.assert_almost_equal(precisions, desired_precisions,
                                          decimal=2)
        numpy.testing.assert_almost_equal(recalls, desired_recalls,
                                          decimal=2)
        numpy.testing.assert_almost_equal(fs, desired_fs,
                                          decimal=2)

    def test_fbeta_multiclass_with_mask(self):
        # [0, 1, 1, 1, 3, 1]
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        mask = torch.Tensor([1, 1, 1, 1, 1, 0])

        fbeta = FBetaMeasure()
        fbeta(predictions, targets, mask)
        precisions, recalls, fs = fbeta.get_metric()

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 3, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [2, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._tp_sum.tolist(), [1, 1, 0, 1, 0])

        desired_precisions = [1.00, 0.33, 0.00, 1.00, 0.00]
        desired_recalls = [0.50, 1.00, 0.00, 1.00, 0.00]
        desired_fs = [(2*p*r)/(p+r) if p+r != 0.0 else 0.0
                      for p, r in zip(desired_precisions, desired_recalls)]
        numpy.testing.assert_almost_equal(precisions, desired_precisions,
                                          decimal=2)
        numpy.testing.assert_almost_equal(recalls, desired_recalls,
                                          decimal=2)
        numpy.testing.assert_almost_equal(fs, desired_fs,
                                          decimal=2)

    def test_fbeta_multiclass_marco_average(self):
        # [0, 1, 1, 1, 3, 1]
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        fbeta = FBetaMeasure(average='macro')
        fbeta(predictions, targets)
        precisions, recalls, fs = fbeta.get_metric()

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 4, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [3, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._tp_sum.tolist(), [1, 1, 0, 1, 0])

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00]
        desired_fs = [(2*p*r)/(p+r) if p+r != 0.0 else 0.0
                      for p, r in zip(desired_precisions, desired_recalls)]
        macro_precision = numpy.mean(desired_precisions)
        macro_recall = numpy.mean(desired_recalls)
        macro_fscore = numpy.mean(desired_fs)
        numpy.testing.assert_almost_equal(precisions, macro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recalls, macro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fs, macro_fscore, decimal=2)

    def test_fbeta_multiclass_micro_average(self):
        # [0, 1, 1, 1, 3, 1]
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        fbeta = FBetaMeasure(average='micro')
        fbeta(predictions, targets)
        precision, recall, fscore = fbeta.get_metric()

        numpy.testing.assert_almost_equal(fbeta._pred_sum.tolist(), [1, 4, 0, 1, 0])
        numpy.testing.assert_almost_equal(fbeta._true_sum.tolist(), [3, 1, 0, 1, 1])
        numpy.testing.assert_almost_equal(fbeta._tp_sum.tolist(), [1, 1, 0, 1, 0])

        true_positives = [1, 1, 0, 1, 0]
        false_positives = [0, 3, 0, 0, 0]
        false_negatives = [2, 0, 0, 0, 1]
        mean_true_positive = numpy.mean(true_positives)
        mean_false_positive = numpy.mean(false_positives)
        mean_false_negative = numpy.mean(false_negatives)

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        numpy.testing.assert_almost_equal(precision, micro_precision, decimal=2)
        numpy.testing.assert_almost_equal(recall, micro_recall, decimal=2)
        numpy.testing.assert_almost_equal(fscore, micro_fscore, decimal=2)

    def test_fbeta_multiclass_with_labels(self):
        # same prediction but with and explicit label ordering
        # [0, 1, 1, 1, 3, 1]
        predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0],
                                    [0.1, 0.5, 0.1, 0.2, 0.0],
                                    [0.1, 0.2, 0.1, 0.7, 0.0],
                                    [0.1, 0.6, 0.1, 0.2, 0.0]])
        targets = torch.Tensor([0, 4, 1, 0, 3, 0])

        fbeta = FBetaMeasure(labels=[4, 3, 2, 1, 0])
        fbeta(predictions, targets)
        precisions, recalls, fs = fbeta.get_metric()

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00][::-1]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00][::-1]
        desired_fs = [(2*p*r)/(p+r) if p+r != 0.0 else 0.0
                      for p, r in zip(desired_precisions, desired_recalls)]
        numpy.testing.assert_almost_equal(precisions, desired_precisions,
                                          decimal=2)
        numpy.testing.assert_almost_equal(recalls, desired_recalls,
                                          decimal=2)
        numpy.testing.assert_almost_equal(fs, desired_fs,
                                          decimal=2)
