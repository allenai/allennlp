import numpy
import pytest
from _pytest.monkeypatch import MonkeyPatch
from transformers.modeling_bert import BertConfig, BertModel
from transformers.tokenization_bert import BertTokenizer

from allennlp.common.testing import ModelTestCase
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.dataset_readers.dataset_utils.span_utils import to_bioul


class BertSrlTest(ModelTestCase):
    def setUp(self):

        self.monkeypatch = MonkeyPatch()
        # monkeypatch the PretrainedBertModel to return the tiny test fixture model
        config_path = self.FIXTURES_ROOT / "bert" / "config.json"
        vocab_path = self.FIXTURES_ROOT / "bert" / "vocab.txt"
        config = BertConfig.from_json_file(config_path)
        self.monkeypatch.setattr(BertModel, "from_pretrained", lambda _: BertModel(config))
        self.monkeypatch.setattr(
            BertTokenizer, "from_pretrained", lambda _: BertTokenizer(vocab_path)
        )

        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "bert_srl" / "experiment.jsonnet",
            self.FIXTURES_ROOT / "conll_2012",
        )

    def tearDown(self):
        self.monkeypatch.undo()
        self.monkeypatch.undo()
        super().tearDown()

    def test_bert_srl_model_can_train_save_and_load(self):
        ignore_grads = {"bert_model.pooler.dense.weight", "bert_model.pooler.dense.bias"}
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=ignore_grads)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        class_probs = output_dict["class_probabilities"][0].data.numpy()
        numpy.testing.assert_almost_equal(
            numpy.sum(class_probs, -1), numpy.ones(class_probs.shape[0]), decimal=6
        )

    @pytest.mark.skip("test-install fails on this test in some environments")
    def test_decode_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)
        lengths = get_lengths_from_binary_sequence_mask(decode_output_dict["mask"]).data.tolist()
        # Hard to check anything concrete which we haven't checked in the above
        # test, so we'll just check that the tags are equal to the lengths
        # of the individual instances, rather than the max length.
        for prediction, length in zip(decode_output_dict["wordpiece_tags"], lengths):
            assert len(prediction) == length

        for prediction, length in zip(decode_output_dict["tags"], lengths):
            # to_bioul throws an exception if the tag sequence is not well formed,
            # so here we can easily check that the sequence we produce is good.
            to_bioul(prediction, encoding="BIO")
