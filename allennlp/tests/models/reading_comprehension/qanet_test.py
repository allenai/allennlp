# pylint: disable=no-self-use,invalid-name
import torch
import pytest
from flaky import flaky
import numpy
from numpy.testing import assert_almost_equal

from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer


class QaNetTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'qanet' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'squad.json')

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)

        metrics = self.model.get_metrics(reset=True)
        # We've set up the data such that there's a fake answer that consists of the whole
        # paragraph.  _Any_ valid prediction for that question should produce an F1 of greater than
        # zero, while if we somehow haven't been able to load the evaluation data, or there was an
        # error with using the evaluation script, this will fail.  This makes sure that we've
        # loaded the evaluation data correctly and have hooked things up to the official evaluation
        # script.
        assert metrics['f1'] > 0

        span_start_probs = output_dict['span_start_probs'][0].data.numpy()
        span_end_probs = output_dict['span_start_probs'][0].data.numpy()
        assert_almost_equal(numpy.sum(span_start_probs, -1), 1, decimal=6)
        assert_almost_equal(numpy.sum(span_end_probs, -1), 1, decimal=6)
        span_start, span_end = tuple(output_dict['best_span'][0].data.numpy())
        assert span_start >= 0
        assert span_start <= span_end
        assert span_end < self.instances[0].fields['passage'].sequence_length()
        assert isinstance(output_dict['best_span_str'][0], str)

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="Need multiple GPUs.")
    def test_multigpu_qanet(self):
        params = Params.from_file(self.param_file)
        vocab = Vocabulary.from_instances(self.instances)
        model = Model.from_params(vocab=vocab, params=params['model']).cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), 0.01, momentum=0.9)
        multigpu_iterator = BasicIterator(batch_size=4)
        multigpu_iterator.index_with(model.vocab)
        trainer = Trainer(model,
                          optimizer,
                          multigpu_iterator,
                          self.instances,
                          num_epochs=2,
                          cuda_device=[0, 1])
        trainer.train()

    def test_batch_predictions_are_consistent(self):
        # The same issue as the bidaf test case.
        # The CNN encoder has problems with this kind of test - it's not properly masked yet, so
        # changing the amount of padding in the batch will result in small differences in the
        # output of the encoder. So, we'll remove the CNN encoder entirely from the model for this test.
        # Save some state.
        # pylint: disable=protected-access,attribute-defined-outside-init
        saved_model = self.model
        saved_instances = self.instances

        # Modify the state, run the test with modified state.
        params = Params.from_file(self.param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])
        reader._token_indexers = {'tokens': reader._token_indexers['tokens']}
        self.instances = reader.read(self.FIXTURES_ROOT / 'data' / 'squad.json')
        vocab = Vocabulary.from_instances(self.instances)
        for instance in self.instances:
            instance.index_fields(vocab)
        del params['model']['text_field_embedder']['token_embedders']['token_characters']
        params['model']['phrase_layer']['num_convs_per_block'] = 0
        params['model']['modeling_layer']['num_convs_per_block'] = 0
        self.model = Model.from_params(vocab=vocab, params=params['model'])

        self.ensure_batch_predictions_are_consistent()

        # Restore the state.
        self.model = saved_model
        self.instances = saved_instances
