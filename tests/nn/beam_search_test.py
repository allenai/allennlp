from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import BeamSearch


transition_probabilities = torch.tensor(
    [
        [0.0, 0.4, 0.3, 0.2, 0.1, 0.0],  # start token -> jth token
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 1st token -> jth token
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2nd token -> jth token
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # ...
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # ...
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]  # end token -> jth token
)


def take_step(
    last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Take decoding step.

    This is a simple function that defines how probabilities are computed for the
    next time step during the beam search.

    We use a simple target vocabulary of size 6. In this vocabulary, index 0 represents
    the start token, and index 5 represents the end token. The transition probability
    from a state where the last predicted token was token `j` to new token `i` is
    given by the `(i, j)` element of the matrix `transition_probabilities`.
    """
    log_probs_list = []
    for last_token in last_predictions:
        log_probs = torch.log(transition_probabilities[last_token.item()])
        log_probs_list.append(log_probs)

    return torch.stack(log_probs_list), state


class BeamSearchTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.end_index = transition_probabilities.size()[0] - 1
        self.beam_search = BeamSearch(self.end_index, max_steps=10, beam_size=3)

        # This is what the top k should look like for each item in the batch.
        self.expected_top_k = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 5], [3, 4, 5, 5, 5]])

        # This is what the log probs should look like for each item in the batch.
        self.expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))

    def _check_results(
        self,
        batch_size: int = 5,
        expected_top_k: np.array = None,
        expected_log_probs: np.array = None,
        beam_search: BeamSearch = None,
        state: Dict[str, torch.Tensor] = None,
    ) -> None:
        expected_top_k = expected_top_k if expected_top_k is not None else self.expected_top_k
        expected_log_probs = (
            expected_log_probs if expected_log_probs is not None else self.expected_log_probs
        )
        state = state or {}

        beam_search = beam_search or self.beam_search
        beam_size = beam_search.beam_size

        initial_predictions = torch.tensor([0] * batch_size)
        top_k, log_probs = beam_search.search(initial_predictions, state, take_step)  # type: ignore

        # top_k should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(top_k.size())[:-1] == [batch_size, beam_size]
        np.testing.assert_array_equal(top_k[0].numpy(), expected_top_k)

        # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(log_probs.size()) == [batch_size, beam_size]
        np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs)

    def test_search(self):
        self._check_results()

    def test_finished_state(self):
        state = {}
        state["foo"] = torch.tensor([[1, 0, 1], [2, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
        # shape: (batch_size, 3)

        expected_finished_state = {}
        expected_finished_state["foo"] = np.array(
            [
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [2, 0, 1],
                [2, 0, 1],
                [2, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        # shape: (batch_size x beam_size, 3)

        self._check_results(state=state)

        # check finished state.
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_diff_shape_state(self):
        state = {}
        state["decoder_hidden"] = torch.tensor(
            [[1, 0, 1], [2, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
        )
        state["decoder_hidden"] = state["decoder_hidden"].unsqueeze(0).repeat(2, 1, 1)
        # shape: (2, batch_size, 3)

        seq = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        seq = [seq] * 2
        expected_finished_state = {}
        expected_finished_state["decoder_hidden"] = np.array(seq)
        # shape: (2, batch_size x beam_size, 3)

        self._check_results(state=state)

        # check finished state.
        for key, array in expected_finished_state.items():
            np.testing.assert_allclose(state[key].numpy(), array)

    def test_batch_size_of_one(self):
        self._check_results(batch_size=1)

    def test_greedy_search(self):
        beam_search = BeamSearch(self.end_index, beam_size=1)
        expected_top_k = np.array([[1, 2, 3, 4, 5]])
        expected_log_probs = np.log(np.array([0.4]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            beam_search=beam_search,
        )

    def test_early_stopping(self):
        """
        Checks case where beam search will reach `max_steps` before finding end tokens.
        """
        beam_search = BeamSearch(self.end_index, beam_size=3, max_steps=3)
        expected_top_k = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            beam_search=beam_search,
        )

    def test_different_per_node_beam_size(self):
        # per_node_beam_size = 1
        beam_search = BeamSearch(self.end_index, beam_size=3, per_node_beam_size=1)
        self._check_results(beam_search=beam_search)

        # per_node_beam_size = 2
        beam_search = BeamSearch(self.end_index, beam_size=3, per_node_beam_size=2)
        self._check_results(beam_search=beam_search)

    def test_catch_bad_config(self):
        """
        If `per_node_beam_size` (which defaults to `beam_size`) is larger than
        the size of the target vocabulary, `BeamSearch.search` should raise
        a ConfigurationError.
        """
        beam_search = BeamSearch(self.end_index, beam_size=20)
        with pytest.raises(ConfigurationError):
            self._check_results(beam_search=beam_search)

    def test_warn_for_bad_log_probs(self):
        # The only valid next step from the initial predictions is the end index.
        # But with a beam size of 3, the call to `topk` to find the 3 most likely
        # next beams will result in 2 new beams that are invalid, in that have probability of 0.
        # The beam search should warn us of this.
        initial_predictions = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        with pytest.warns(RuntimeWarning, match="Infinite log probabilities"):
            self.beam_search.search(initial_predictions, {}, take_step)

    def test_empty_sequences(self):
        initial_predictions = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        beam_search = BeamSearch(self.end_index, beam_size=1)
        with pytest.warns(RuntimeWarning, match="Empty sequences predicted"):
            predictions, log_probs = beam_search.search(initial_predictions, {}, take_step)
        # predictions hould have shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(predictions.size()) == [2, 1, 1]
        # log probs hould have shape `(batch_size, beam_size)`.
        assert list(log_probs.size()) == [2, 1]
        assert (predictions == self.end_index).all()
        assert (log_probs == 0).all()
