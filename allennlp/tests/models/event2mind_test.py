# pylint: disable=invalid-name
from allennlp.commands.train import train_model_from_file
from allennlp.common.testing import ModelTestCase
from allennlp.nn.util import get_text_field_mask


class Event2MindTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / "event2mind" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "event2mind_medium.csv")

    #def test_encoder_decoder_can_train_save_and_load(self):
    #    self.ensure_model_can_train_save_and_load(self.param_file)

    def test_beam_search_matches_greedy(self):
        save_dir = self.TEST_DIR / "beam_search_test"
        archive_file = save_dir / "model.tar.gz"
        model = train_model_from_file(self.param_file, save_dir)

        for instance in self.dataset:
            cur_text_field = instance.fields["source"]
            text = [token.text for token in cur_text_field.tokens]
            if text == ["@start@", "personx", "calls", "personx", "'s", "brother", "@end@"]:
              desired_text_field = cur_text_field
              break
        print(desired_text_field)
        source = desired_text_field.as_tensor(desired_text_field.get_padding_lengths())
        source['tokens'] = source['tokens'].unsqueeze(0)
        print(source)
        embedded_input = model._embedding_dropout(model._source_embedder(source))
        source_mask = get_text_field_mask(source)
        final_encoder_output = model._encoder(embedded_input, source_mask)

        state = model._states["xintent"]
        greedy_prediction = model.greedy_predict(final_encoder_output,
                                                 state.embedder,
                                                 state.decoder_cell,
                                                 state.output_projection_layer)
        print(greedy_prediction.size())
        greedy_tokens = model.decode_all(greedy_prediction)
        print(greedy_tokens)

        (beam_predictions, _) = model.beam_search(
                final_encoder_output,
                1,
                model._max_decoding_steps,
                state.embedder,
                state.decoder_cell,
                state.output_projection_layer
        )
        beam_prediction = beam_predictions[0]
        print(beam_prediction.size())
        beam_tokens = model.decode_all(beam_prediction)
        print(beam_tokens)

        raise Exception("fdskjfkdj")
