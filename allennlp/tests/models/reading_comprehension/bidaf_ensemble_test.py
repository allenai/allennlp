# pylint: disable=no-self-use,invalid-name
import numpy
import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.models.reading_comprehension.bidaf_ensemble import BidafEnsemble, ensemble

class BidafEnsembleTest(ModelTestCase):
    def setUp(self):
        super(BidafEnsembleTest, self).setUp()
        self.set_up_model('tests/fixtures/bidaf/experiment.json', 'tests/fixtures/data/squad.json')
        self.model.eval()

    def test_ensemble_chooses_highest_average_confidence_2(self):
        subresults = [
                {
                        "span_start_probs": Variable(torch.FloatTensor([[0.9, 0.0, 0.0, 0.0]])),
                        "span_end_probs": Variable(torch.FloatTensor([[0.9, 0.0, 0.0, 0.0]])),
                        "best_span": Variable(torch.LongTensor([[0, 0]])),
                        "best_span_str": "What",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"],
                        "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": Variable(torch.FloatTensor([[0.0, 0.0, 1.0, 0.0]])),
                        "span_end_probs": Variable(torch.FloatTensor([[0.0, 0.0, 1.0, 0.0]])),
                        "best_span": Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"],
                        "passage_tokens": ["Michael", "ate", "cheese", "."]
                }
        ]

        numpy.testing.assert_almost_equal(
                ensemble(subresults).data[0].cpu().numpy(),
                torch.LongTensor([2, 2]).cpu().numpy())

    def test_ensemble_chooses_highest_average_confidence_3(self):
        subresults = [
                {
                        "span_start_probs": Variable(torch.FloatTensor([[0.0, 0.0, 0.9, 0.1]])),
                        "span_end_probs": Variable(torch.FloatTensor([[0.0, 0.0, 0.9, 0.1]])),
                        "best_span": Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"],
                        "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": Variable(torch.FloatTensor([[0.0, 0.0, 0.9, 0.1]])),
                        "span_end_probs": Variable(torch.FloatTensor([[0.0, 0.0, 0.9, 0.1]])),
                        "best_span": Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"],
                        "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": Variable(torch.FloatTensor([[0.9, 0.0, 0.0, 0.0]])),
                        "span_end_probs": Variable(torch.FloatTensor([[0.9, 0.0, 0.0, 0.0]])),
                        "best_span": Variable(torch.LongTensor([[0, 0]])),
                        "best_span_str": "What",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"],
                        "passage_tokens": ["Michael", "ate", "cheese", "."]
                }
        ]

        numpy.testing.assert_almost_equal(
                ensemble(subresults).data[0].cpu().numpy(),
                torch.LongTensor([2, 2]).numpy())

    def test_forward_pass_runs_correctly(self):
        """
        Check to make sure a forward pass on an ensemble of two identical copies of a model yields the same
        results as the model itself.
        """
        bidaf_ensemble = BidafEnsemble([self.model, self.model])

        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()

        bidaf_output_dict = self.model(**training_tensors)
        ensemble_output_dict = bidaf_ensemble(**training_tensors)

        metrics = self.model.get_metrics(reset=True)

        # We've set up the data such that there's a fake answer that consists of the whole
        # paragraph.  _Any_ valid prediction for that question should produce an F1 of greater than
        # zero, while if we somehow haven't been able to load the evaluation data, or there was an
        # error with using the evaluation script, this will fail.  This makes sure that we've
        # loaded the evaluation data correctly and have hooked things up to the official evaluation
        # script.
        assert metrics['f1'] > 0
        assert torch.equal(ensemble_output_dict['best_span'], bidaf_output_dict['best_span'])
        assert ensemble_output_dict['best_span_str'] == bidaf_output_dict['best_span_str']
