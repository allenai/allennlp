# pylint: disable=no-self-use,invalid-name,protected-access

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSeq2SeqPredictor(AllenNlpTestCase):
    def test_uses_named_inputs_with_simple_seq2seq(self):
        inputs = {
                "source": "What kind of test succeeded on its first attempt?",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'encoder_decoder' / 'simple_seq2seq' /
                               'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'seq2seq')

        result = predictor.predict_json(inputs)

        predicted_tokens = result.get("predicted_tokens")
        assert predicted_tokens is not None
        assert isinstance(predicted_tokens, list)
        assert all(isinstance(x, str) for x in predicted_tokens)

    def test_copynet_predictions(self):
        archive = load_archive(self.FIXTURES_ROOT / 'encoder_decoder' / 'copynet_seq2seq' /
                               'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'seq2seq')
        model = predictor._model
        end_token = model.vocab.get_token_from_index(model._end_index, model._target_namespace)
        output_dict = predictor.predict("these tokens should be copied over : hello world")
        assert len(output_dict["predictions"]) == model._beam_search.beam_size
        assert len(output_dict["predicted_tokens"]) == model._beam_search.beam_size
        for predicted_tokens in output_dict["predicted_tokens"]:
            assert all(isinstance(x, str) for x in predicted_tokens)
            assert end_token not in predicted_tokens
