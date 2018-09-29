# pylint: disable=invalid-name,protected-access
import torch

from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_text_field_mask


class Event2MindTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / "event2mind" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "event2mind_medium.csv")
        save_dir = self.TEST_DIR / "trained_model_tests"
        self.trained_model = train_model_from_file(self.param_file, save_dir)

    def test_encoder_decoder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def get_sample_encoded_output(self):
        """
        Returns the encoded vector for a sample event.
        """
        for instance in self.dataset:
            cur_text_field = instance.fields["source"]
            text = [token.text for token in cur_text_field.tokens]
            if text == ["@start@", "personx", "calls", "personx", "'s", "brother", "@end@"]:
                sample_text_field = cur_text_field
                break
        source = sample_text_field.as_tensor(sample_text_field.get_padding_lengths())
        source['tokens'] = source['tokens'].unsqueeze(0)
        embedded_input = self.trained_model._embedding_dropout(
                self.trained_model._source_embedder(source)
        )
        source_mask = get_text_field_mask(source)
        return self.trained_model._encoder(embedded_input, source_mask)

    def test_beam_search_orders_results(self):
        model = self.trained_model
        state = model._states["xintent"]

        final_encoder_output = self.get_sample_encoded_output()
        batch_size = final_encoder_output.size()[0]
        start_predictions = final_encoder_output.new_full(
                (batch_size,), fill_value=model._start_index, dtype=torch.long)
        start_state = {"decoder_hidden": final_encoder_output}

        # (batch_size, 10, num_decoding_steps)
        _, batch_logits = model._beam_search.search(
                start_predictions, start_state, state.take_step)

        logits = batch_logits[0]
        # Sanity check beam size.
        assert logits.size()[0] == 10
        prev_logit = 0
        for cur_logit in logits:
            assert cur_logit <= prev_logit
            prev_logit = cur_logit

    def test_beam_search_matches_greedy(self):
        model = self.trained_model
        state = model._states["xintent"]
        beam_search = BeamSearch(model._end_index,
                                 max_steps=model._max_decoding_steps,
                                 beam_size=1)

        final_encoder_output = self.get_sample_encoded_output()
        batch_size = final_encoder_output.size()[0]
        start_predictions = final_encoder_output.new_full(
                (batch_size,), fill_value=model._start_index, dtype=torch.long)
        start_state = {"decoder_hidden": final_encoder_output}

        greedy_prediction = model.greedy_predict(
                final_encoder_output=final_encoder_output,
                target_embedder=state.embedder,
                decoder_cell=state.decoder_cell,
                output_projection_layer=state.output_projection_layer
        )
        greedy_tokens = model.decode_all(greedy_prediction)

        (beam_predictions, _) = beam_search.search(
                start_predictions, start_state, state.take_step)
        beam_prediction = beam_predictions[0]
        beam_tokens = model.decode_all(beam_prediction)

        assert beam_tokens == greedy_tokens
