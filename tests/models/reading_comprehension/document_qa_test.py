# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch


class DocumentQaTriviaQaPreprocessedTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
                'tests/fixtures/document_qa/triviaqa.processed.json',
                'web-train.jsonl'
        )

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()

        output_dict = self.model(**training_tensors)
        metrics = self.model.get_metrics(reset=True)

        assert 'span_start_logits' in output_dict
        assert 'best_span' in output_dict
        assert 'loss' in output_dict

        assert 'em' in metrics
        assert 'f1' in metrics

class DocumentQaTriviaQaTarGzTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
                'tests/fixtures/document_qa/triviaqa.json',
                'web-train.json'
        )

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()

        output_dict = self.model(**training_tensors)
        metrics = self.model.get_metrics(reset=True)

        assert 'span_start_logits' in output_dict
        assert 'best_span' in output_dict
        assert 'loss' in output_dict

        assert 'em' in metrics
        assert 'f1' in metrics


class DocumentQaSquadTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
                'tests/fixtures/document_qa/squad.json',
                'tests/fixtures/data/squad.json'
        )

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()

        output_dict = self.model(**training_tensors)
        metrics = self.model.get_metrics(reset=True)

        assert 'span_start_logits' in output_dict
        assert 'best_span' in output_dict
        assert 'loss' in output_dict

        assert 'em' in metrics
        assert 'f1' in metrics
