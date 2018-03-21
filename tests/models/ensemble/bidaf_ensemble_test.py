# pylint: disable=no-self-use,invalid-name,line-too-long
import torch

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.models.ensemble.bidaf_ensemble import BidafEnsemble, ensemble

class BidafEnsembleTest(ModelTestCase):
    def setUp(self):
        super(BidafEnsembleTest, self).setUp()
        self.set_up_model('tests/fixtures/bidaf/experiment.json', 'tests/fixtures/data/squad.json')
        self.model.eval()

    def test_ensemble_chooses_most_votes(self):
        subresults = [
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.0009347021696157753, 0.0012768324231728911, 0.9999938261222839, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[4.180105679552071e-05, 2.762122494459618e-05, 0.9999965809440613, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.0009347021696157753, 0.0012768324231728911, 0.9977738261222839, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[4.180105679552071e-05, 2.762122494459618e-05, 0.9158865809440613, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.9977738261222839, 0.0009347021696157753, 0.0012768324231728911, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[0.9158865809440613, 4.180105679552071e-05, 2.762122494459618e-05, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[0, 0]])),
                        "best_span_str": "What",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                }
        ]

        assert ensemble(0, subresults) == 0

    def test_ensemble_chooses_highest_average_confidence(self):
        subresults = [
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.0009347021696157753, 0.0012768324231728911, 0.9977738261222839, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[4.180105679552071e-05, 2.762122494459618e-05, 0.9158865809440613, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.9977738261222839, 0.0009347021696157753, 0.0012768324231728911, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[0.9158865809440613, 4.180105679552071e-05, 2.762122494459618e-05, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[0, 0]])),
                        "best_span_str": "What",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                }
        ]

        assert ensemble(0, subresults) == 0

    def test_ensemble_chooses_highest_average_confidence_reverse(self):
        subresults = [
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.1117738261222839, 0.0009347021696157753, 0.0012768324231728911, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[0.1118865809440613, 4.180105679552071e-05, 2.762122494459618e-05, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[0, 0]])),
                        "best_span_str": "What",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
                {
                        "span_start_probs": torch.autograd.Variable(torch.FloatTensor([[0.0009347021696157753, 0.0012768324231728911, 0.9977738261222839, 1.4630706573370844e-05]])),
                        "span_end_probs": torch.autograd.Variable(torch.FloatTensor([[4.180105679552071e-05, 2.762122494459618e-05, 0.9158865809440613, 0.0840439721941948]])),
                        "best_span": torch.autograd.Variable(torch.LongTensor([[2, 2]])),
                        "best_span_str": "cheese",
                        "question_tokens": ["What", "did", "Michael", "eat", "?"], "passage_tokens": ["Michael", "ate", "cheese", "."]
                },
        ]

        assert ensemble(0, subresults) == 1

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
        assert torch.equal(ensemble_output_dict['best_span'], bidaf_output_dict['best_span'].data)
        assert ensemble_output_dict['best_span_str'] == bidaf_output_dict['best_span_str']
