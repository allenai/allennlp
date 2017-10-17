# pylint: disable=no-self-use,invalid-name

import torch
from torch.autograd import Variable
import pytest
from allennlp.common.testing import ModelTestCase


class CorefTest(ModelTestCase):
    def setUp(self):
        super(CorefTest, self).setUp()
        self.set_up_model('tests/fixtures/coref/experiment.json',
                          'tests/fixtures/data/coref/sample.gold_conll')

    def test_coref_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @pytest.mark.skip
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_decode(self):

        spans = torch.LongTensor([[1, 2],
                                  [3, 4],
                                  [3, 7],
                                  [5, 6],
                                  [14, 56],
                                  [17, 80]])
        spans = Variable(spans.unsqueeze(0))

        # Indices into ``spans`` indicating that the two mentions
        # are co-referent.
        antecedents = torch.LongTensor([-1, 0, -1, -1, 1, 3])
        antecedents = Variable(antecedents.unsqueeze(0))
        output_dict = {
                        "top_spans": spans,
                        "predicted_antecedents": antecedents
                    }
        output = self.model.decode(output_dict)

        clusters = output["clusters"][0]
        gold1 = [(1, 2), (3, 4), (14, 56)]
        gold2 = [(5, 6), (17, 80)]
        assert gold1 in clusters
        clusters.remove(gold1)
        assert gold2 in clusters
        clusters.remove(gold2)
        assert clusters == []
