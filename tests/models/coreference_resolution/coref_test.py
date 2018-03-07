# pylint: disable=no-self-use,invalid-name

import torch
from torch.autograd import Variable
from allennlp.common.testing import ModelTestCase


class CorefTest(ModelTestCase):
    def setUp(self):
        super(CorefTest, self).setUp()
        self.set_up_model('tests/fixtures/coref/experiment.json',
                          'tests/fixtures/coref/coref.gold_conll')

    def test_coref_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_decode(self):

        spans = torch.LongTensor([[1, 2],
                                  [3, 4],
                                  [3, 7],
                                  [5, 6],
                                  [14, 56],
                                  [17, 80]])

        antecedent_indices = torch.LongTensor([[0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0],
                                               [1, 0, 0, 0, 0, 0],
                                               [2, 1, 0, 0, 0, 0],
                                               [3, 2, 1, 0, 0, 0],
                                               [4, 3, 2, 1, 0, 0]])

        spans = Variable(spans.unsqueeze(0))
        antecedent_indices = Variable(antecedent_indices)
        # Indices into ``antecedent_indices`` indicating the predicted antecedent
        # index in ``top_spans``.
        predicted_antecedents = torch.LongTensor([-1, 0, -1, -1, 1, 3])
        predicted_antecedents = Variable(predicted_antecedents.unsqueeze(0))
        output_dict = {
                "top_spans": spans,
                "antecedent_indices": antecedent_indices,
                "predicted_antecedents": predicted_antecedents
                }
        output = self.model.decode(output_dict)

        clusters = output["clusters"][0]
        gold1 = [(1, 2), (3, 4), (17, 80)]
        gold2 = [(3, 7), (14, 56)]

        assert len(clusters) == 2
        assert gold1 in clusters
        assert gold2 in clusters
