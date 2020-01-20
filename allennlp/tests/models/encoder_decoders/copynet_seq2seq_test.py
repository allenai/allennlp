import numpy as np
from scipy.special import logsumexp
import torch

from allennlp.common.testing import ModelTestCase


class CopyNetTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / "encoder_decoder" / "copynet_seq2seq" / "experiment.json",
            self.FIXTURES_ROOT / "data" / "copynet" / "copyover.tsv",
        )

    def test_model_can_train_save_load_predict(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_vocab(self):
        vocab = self.model.vocab
        assert vocab.get_vocab_size(self.model._target_namespace) == 8
        assert "hello" not in vocab._token_to_index[self.model._target_namespace]
        assert "world" not in vocab._token_to_index[self.model._target_namespace]
        assert "@COPY@" in vocab._token_to_index["target_tokens"]

    def test_train_instances(self):
        inputs = self.instances[0].as_tensor_dict()
        source_tokens = inputs["source_tokens"]["tokens"]
        target_tokens = inputs["target_tokens"]["tokens"]

        assert list(source_tokens["tokens"].size()) == [11]
        assert list(target_tokens["tokens"].size()) == [10]

        assert target_tokens["tokens"][0] == self.model._start_index
        assert target_tokens["tokens"][4] == self.model._oov_index
        assert target_tokens["tokens"][5] == self.model._oov_index
        assert target_tokens["tokens"][-1] == self.model._end_index

    def test_get_ll_contrib(self):
        # batch_size = 3, trimmed_input_len = 3
        #
        # In the first instance, the contribution to the likelihood should
        # come from both the generation scores and the copy scores, since the
        # token is in the source sentence and the target vocabulary.
        # In the second instance, the contribution should come only from the
        # generation scores, since the token is not in the source sentence.
        # In the third instance, the contribution should come only from the copy scores,
        # since the token is in the source sequence but is not in the target vocabulary.

        vocab = self.model.vocab

        generation_scores = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # these numbers are arbitrary.
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        # shape: (batch_size, target_vocab_size)

        copy_scores = torch.tensor(
            [[1.0, 2.0, 1.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]  # these numbers are arbitrary.
        )
        # shape: (batch_size, trimmed_input_len)

        target_tokens = torch.tensor(
            [
                vocab.get_token_index("tokens", self.model._target_namespace),
                vocab.get_token_index("the", self.model._target_namespace),
                self.model._oov_index,
            ]
        )
        # shape: (batch_size,)

        target_to_source = torch.tensor([[0, 1, 0], [0, 0, 0], [1, 0, 1]])
        # shape: (batch_size, trimmed_input_len)

        copy_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # shape: (batch_size, trimmed_input_len)

        # This is what the log likelihood result should look like.
        ll_check = np.array(
            [
                # First instance.
                logsumexp(np.array([generation_scores[0, target_tokens[0].item()].item(), 2.0]))
                - logsumexp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0])),
                # Second instance.
                generation_scores[1, target_tokens[1].item()].item()
                - logsumexp(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])),
                # Third instance.
                logsumexp(np.array([2.0, 3.0]))
                - logsumexp(np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0, 3.0])),
            ]
        )

        # This is what the selective_weights result should look like.
        selective_weights_check = np.stack(
            [
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.exp([2.0, float("-inf"), 3.0]) / (np.exp(2.0) + np.exp(3.0)),
            ]
        )

        generation_scores_mask = generation_scores.new_full(generation_scores.size(), 1.0)
        ll_actual, selective_weights_actual = self.model._get_ll_contrib(
            generation_scores,
            generation_scores_mask,
            copy_scores,
            target_tokens,
            target_to_source,
            copy_mask,
        )

        np.testing.assert_almost_equal(ll_actual.data.numpy(), ll_check, decimal=6)

        np.testing.assert_almost_equal(
            selective_weights_actual.data.numpy(), selective_weights_check, decimal=6
        )

    def test_get_input_and_selective_weights(self):
        target_vocab_size = self.model._target_vocab_size
        oov_index = self.model._oov_index
        copy_index = self.model._copy_index

        # shape: (group_size,)
        last_predictions = torch.tensor(
            [5, 6, target_vocab_size + 1]  # only generated.  # copied AND generated.
        )  # only copied.
        # shape: (group_size, trimmed_source_length)
        source_to_target = torch.tensor(
            [[6, oov_index, oov_index], [6, oov_index, 6], [5, oov_index, oov_index]]
        )
        # shape: (group_size, trimmed_source_length)
        source_token_ids = torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 0],
                [0, 1, 1],
            ]  # no duplicates.  # first and last source tokens match.
        )  # middle and last source tokens match.
        # shape: (group_size, trimmed_source_length)
        copy_probs = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

        state = {
            "source_to_target": source_to_target,
            "source_token_ids": source_token_ids,
            "copy_log_probs": (copy_probs + 1e-45).log(),
        }

        input_choices, selective_weights = self.model._get_input_and_selective_weights(
            last_predictions, state
        )
        assert list(input_choices.size()) == [3]
        assert list(selective_weights.size()) == [3, 3]

        # shape: (group_size,)
        input_choices_check = np.array([5, 6, copy_index])
        np.testing.assert_equal(input_choices.numpy(), input_choices_check)

        # shape: (group_size, trimmed_source_length)
        selective_weights_check = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        np.testing.assert_equal(selective_weights.numpy(), selective_weights_check)

    def test_gather_final_log_probs(self):
        target_vocab_size = self.model._target_vocab_size
        assert target_vocab_size == 8

        oov_index = self.model._oov_index
        assert oov_index not in [5, 6]

        # shape: (group_size, trimmed_source_length)
        source_to_target = torch.tensor([[6, oov_index, oov_index], [oov_index, 5, 5]])
        # shape: (group_size, trimmed_source_length)
        source_token_ids = torch.tensor([[0, 1, 1], [0, 1, 1]])
        # shape: (group_size, target_vocab_size)
        generation_probs = torch.tensor([[0.1] * target_vocab_size, [0.1] * target_vocab_size])
        # shape: (group_size, trimmed_source_length)
        copy_probs = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

        state = {"source_to_target": source_to_target, "source_token_ids": source_token_ids}

        final_log_probs = self.model._gather_final_log_probs(
            generation_probs.log(), copy_probs.log(), state
        )
        final_probs = final_log_probs.exp()
        assert list(final_probs.size()) == [2, target_vocab_size + 3]

        final_probs_check = np.array(
            [
                # First copy token matches a source token. So first copy score is added to
                # corresponding generation score.
                # Second and third copy tokens match, so third copy score added to second
                # copy score.
                [
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.1,  # modified generation scores
                    0.0,
                    0.2,
                    0.0,
                ],  # modified copy scores
                # Second and third copy tokens match the same token in target vocab.
                [
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.3,
                    0.1,
                    0.1,  # modified generation scores
                    0.1,
                    0.0,
                    0.0,
                ],  # modified copy scores
            ]
        )
        np.testing.assert_array_almost_equal(final_probs.numpy(), final_probs_check)

    def test_gather_extended_gold_tokens(self):
        vocab_size = self.model._target_vocab_size
        end_index = self.model._end_index
        pad_index = self.model._pad_index
        oov_index = self.model._oov_index
        tok_index = 6  # some other arbitrary token
        assert tok_index not in [end_index, pad_index, oov_index]

        # first sentence tokens:
        #  1: oov but not copied
        #  2: not oov and not copied
        #  3: not copied
        #  4: not copied
        # second sentence tokens:
        #  1: not oov and copied
        #  2: oov and copied
        #  3: not copied
        #  4: not copied

        # shape: (batch_size, target_sequence_length)
        target_tokens = torch.tensor(
            [
                [oov_index, tok_index, end_index, pad_index],
                [tok_index, oov_index, tok_index, end_index],
            ]
        )
        # shape: (batch_size, trimmed_source_length)
        source_token_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 2]])
        # shape: (batch_size, target_sequence_length)
        target_token_ids = torch.tensor([[4, 5, 6, 7], [1, 0, 3, 4]])
        # shape: (batch_size, target_sequence_length)
        result = self.model._gather_extended_gold_tokens(
            target_tokens, source_token_ids, target_token_ids
        )
        # shape: (batch_size, target_sequence_length)
        check = np.array(
            [
                [oov_index, tok_index, end_index, pad_index],
                [tok_index, vocab_size, tok_index, end_index],
            ]
        )
        np.testing.assert_array_equal(result.numpy(), check)

    def test_get_predicted_tokens(self):
        tok_index = self.vocab.get_token_index("tokens", self.model._target_namespace)
        end_index = self.model._end_index
        vocab_size = self.model._target_vocab_size

        # shape: (batch_size, beam_size, max_predicted_length)
        predicted_indices = np.array(
            [
                [
                    [tok_index, vocab_size, vocab_size + 1, end_index],
                    [tok_index, tok_index, tok_index, tok_index],
                ],
                [
                    [tok_index, tok_index, tok_index, end_index],
                    [tok_index, vocab_size + 1, end_index, end_index],
                ],
            ]
        )

        batch_metadata = [
            {"source_tokens": ["hello", "world"]},
            {"source_tokens": ["copynet", "is", "cool"]},
        ]

        predicted_tokens = self.model._get_predicted_tokens(predicted_indices, batch_metadata)
        assert len(predicted_tokens) == 2
        assert len(predicted_tokens[0]) == 2
        assert len(predicted_tokens[1]) == 2
        assert predicted_tokens[0][0] == ["tokens", "hello", "world"]
        assert predicted_tokens[0][1] == ["tokens", "tokens", "tokens", "tokens"]

        predicted_tokens = self.model._get_predicted_tokens(
            predicted_indices, batch_metadata, n_best=1
        )
        assert len(predicted_tokens) == 2
        assert predicted_tokens[0] == ["tokens", "hello", "world"]
        assert predicted_tokens[1] == ["tokens", "tokens", "tokens"]
