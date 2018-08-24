# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pytest
import numpy
from numpy.testing import assert_almost_equal
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.models.reading_comprehension.dialog_qa import DialogQA


class DialogQATest(ModelTestCase):
    def setUp(self):
        super(DialogQATest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'dialog_qa' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'quac_sample.json')

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert "best_span_str" in output_dict and "loss" in output_dict
        assert "followup" in output_dict and "yesno" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-4)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
        #assert_almost_equal(numpy.sum(span_start_probs, -1), 1, decimal=6)
        #assert_almost_equal(numpy.sum(span_end_probs, -1), 1, decimal=6)
        #span_start, span_end = tuple(output_dict['best_span'][0].data.numpy())
        #assert span_start >= 0
        #assert span_start <= span_end
        #assert span_end < self.instances[0].fields['passage'].sequence_length()
        #assert isinstance(output_dict['best_span_str'][0], str)

    # @flaky
    # def test_batch_predictions_are_consistent(self):
    #     # The CNN encoder has problems with this kind of test - it's not properly masked yet, so
    #     # changing the amount of padding in the batch will result in small differences in the
    #     # output of the encoder.  Because BiDAF is so deep, these differences get magnified through
    #     # the network and make this test impossible.  So, we'll remove the CNN encoder entirely
    #     # from the model for this test.  If/when we fix the CNN encoder to work correctly with
    #     # masking, we can change this back to how the other models run this test, with just a
    #     # single line.
    #     # pylint: disable=protected-access,attribute-defined-outside-init
    #
    #     # Save some state.
    #     saved_model = self.model
    #     saved_instances = self.instances
    #
    #     # Modify the state, run the test with modified state.
    #     params = Params.from_file(self.param_file)
    #     reader = DatasetReader.from_params(params['dataset_reader'])
    #     reader._token_indexers = {'tokens': reader._token_indexers['tokens']}
    #     self.instances = reader.read(self.FIXTURES_ROOT / 'data' / 'quac_sample.json')
    #     vocab = Vocabulary.from_instances(self.instances)
    #     for instance in self.instances:
    #         instance.index_fields(vocab)
    #     del params['model']['text_field_embedder']['token_embedders']['token_characters']
    #     params['model']['phrase_layer']['input_size'] = 2
    #     self.model = Model.from_params(vocab=vocab, params=params['model'])
    #
    #     self.ensure_batch_predictions_are_consistent()
    #
    #     # Restore the state.
    #     self.model = saved_model
    #     self.instances = saved_instances

    #def test_get_best_span(self):
        # pylint: disable=protected-access

     #   span_begin_probs = torch.FloatTensor([[0.1, 0.3, 0.05, 0.3, 0.25]]).log()
     #   span_end_probs = torch.FloatTensor([[0.65, 0.05, 0.2, 0.05, 0.05]]).log()

     #   followup_probs = torch.FloatTensor([[0.3, 0.6, 0.1]]).log()
     #   yesno_probs = torch.FloatTensor([[0.1, 0.2, 0.7]]).log

      #  begin_end_idxs = DialogQA.get_best_span(span_begin_probs, span_end_probs, yesno_probs, followup_probs)
      #  assert_almost_equal(begin_end_idxs.data.numpy(), [[0, 0, 1, 2]])


    # def test_mismatching_dimensions_throws_configuration_error(self):
    #     params = Params.from_file(self.param_file)
    #     # Make the phrase layer wrong - it should be 10 to match
    #     # the embedding + char cnn dimensions.
    #     params["model"]["phrase_layer"]["input_size"] = 12
    #     with pytest.raises(ConfigurationError):
    #         Model.from_params(vocab=self.vocab, params=params.pop("model"))
    #
    #     params = Params.from_file(self.param_file)
    #     # Make the modeling layer input_dimension wrong - it should be 40 to match
    #     # 4 * output_dim of the phrase_layer.
    #     params["model"]["phrase_layer"]["input_size"] = 30
    #     with pytest.raises(ConfigurationError):
    #         Model.from_params(vocab=self.vocab, params=params.pop("model"))
    #
    #     params = Params.from_file(self.param_file)
    #     # Make the modeling layer input_dimension wrong - it should be 70 to match
    #     # 4 * phrase_layer.output_dim + 3 * modeling_layer.output_dim.
    #     params["model"]["span_end_encoder"]["input_size"] = 50
    #     with pytest.raises(ConfigurationError):
    #         Model.from_params(vocab=self.vocab, params=params.pop("model"))
