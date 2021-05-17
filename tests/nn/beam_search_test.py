from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.nn.beam_search import (
    MultinomialSampler,
    BeamSearch,
    TopKSampler,
    TopPSampler,
    GumbelSampler,
)
from allennlp.common.params import Params


transition_probabilities = torch.tensor(
    [
        [0.0, 0.4, 0.3, 0.2, 0.1, 0.0],  # start token -> jth token
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 1st token -> jth token
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 2nd token -> jth token
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # ...
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # ...
        [0.2, 0.1, 0.2, 0.2, 0.2, 0.3],
    ]  # end token -> jth token
)

# A transition matrix that favors shorter sequences over longer ones
short_sequence_transition_probabilities = torch.tensor(
    [
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.9],  # start token -> jth token
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.9],  # 1st token -> jth token
        [0.0, 0.0, 0.0, 0.1, 0.0, 0.9],  # 2nd token -> jth token
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],  # ...
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # ...
        [0.2, 0.1, 0.2, 0.2, 0.2, 0.3],
    ]  # end token -> jth token
)

log_probabilities = torch.log(
    torch.tensor([[0.1, 0.3, 0.3, 0.3, 0.0, 0.0], [0.0, 0.0, 0.4, 0.3, 0.2, 0.1]])
)


def take_step_no_timestep(
    last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
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


def take_step_with_timestep(
    last_predictions: torch.Tensor,
    state: Dict[str, torch.Tensor],
    timestep: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return take_step_no_timestep(last_predictions, state)


def take_short_sequence_step(
    last_predictions: torch.Tensor,
    state: Dict[str, torch.Tensor],
    timestep: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Take decoding step.

    This method is the same as `take_step_no_timestep` except it uses the
    `short_sequence_transition_probabilities` transitions instead of `transition_probabilities`
    """
    log_probs_list = []
    for last_token in last_predictions:
        log_probs = torch.log(short_sequence_transition_probabilities[last_token.item()])
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
        take_step=take_step_with_timestep,
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
        np.testing.assert_allclose(log_probs[0].numpy(), expected_log_probs, rtol=1e-6)

    @pytest.mark.parametrize("step_function", [take_step_with_timestep, take_step_no_timestep])
    def test_search(self, step_function):
        self._check_results(take_step=step_function)

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

    def test_single_step(self):
        self.beam_search.max_steps = 1
        expected_top_k = np.array([[1], [2], [3]])
        expected_log_probs = np.log(np.array([0.4, 0.3, 0.2]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
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

    def test_take_short_sequence_step(self):
        """
        Tests to ensure the top-k from the short_sequence_transition_probabilities
        transition matrix is expected
        """
        self.beam_search.beam_size = 5
        expected_top_k = np.array([
            [5, 5, 5, 5, 5],
            [1, 5, 5, 5, 5],
            [1, 2, 5, 5, 5],
            [1, 2, 3, 5, 5],
            [1, 2, 3, 4, 5]
        ])
        expected_log_probs = np.log(np.array([0.9, 0.09, 0.009, 0.0009, 0.0001]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            take_step=take_short_sequence_step,
        )

    def test_min_steps(self):
        """
        Tests to ensure all output sequences are greater than a specified minimum length.
        It uses the `take_short_sequence_step` step function, which favors shorter sequences.
        See `test_take_short_sequence_step`.
        """
        self.beam_search.beam_size = 1

        # An empty sequence is allowed under this step function
        self.beam_search.min_steps = 0
        expected_top_k = np.array([[5]])
        expected_log_probs = np.log(np.array([0.9]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            take_step=take_short_sequence_step,
        )

        self.beam_search.min_steps = 1
        expected_top_k = np.array([[1, 5]])
        expected_log_probs = np.log(np.array([0.09]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            take_step=take_short_sequence_step,
        )

        self.beam_search.min_steps = 2
        expected_top_k = np.array([[1, 2, 5]])
        expected_log_probs = np.log(np.array([0.009]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            take_step=take_short_sequence_step,
        )

        self.beam_search.beam_size = 3
        self.beam_search.min_steps = 2
        expected_top_k = np.array([[1, 2, 5, 5, 5], [1, 2, 3, 5, 5], [1, 2, 3, 4, 5]])
        expected_log_probs = np.log(np.array([0.009, 0.0009, 0.0001]))
        self._check_results(
            expected_top_k=expected_top_k,
            expected_log_probs=expected_log_probs,
            take_step=take_short_sequence_step,
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
            self.beam_search.search(initial_predictions, {}, take_step_no_timestep)

    def test_empty_sequences(self):
        initial_predictions = torch.LongTensor([self.end_index - 1, self.end_index - 1])
        beam_search = BeamSearch(self.end_index, beam_size=1)
        with pytest.warns(RuntimeWarning, match="Empty sequences predicted"):
            predictions, log_probs = beam_search.search(
                initial_predictions, {}, take_step_with_timestep
            )
        # predictions hould have shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(predictions.size()) == [2, 1, 1]
        # log probs hould have shape `(batch_size, beam_size)`.
        assert list(log_probs.size()) == [2, 1]
        assert (predictions == self.end_index).all()
        assert (log_probs == 0).all()

    def test_default_from_params_params(self):
        beam_search = BeamSearch.from_params(Params({"beam_size": 2, "end_index": 7}))
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7

    def test_top_p_search(self):
        initial_predictions = torch.tensor([0] * 5)
        beam_size = 3
        take_step = take_step_with_timestep
        p_sampler = TopPSampler(p=0.8)

        top_p, log_probs = BeamSearch(
            self.end_index, beam_size=beam_size, max_steps=10, sampler=p_sampler
        ).search(initial_predictions, {}, take_step)

        beam_size = beam_size or 1
        batch_size = 5

        # top_p should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(top_p.size())[:-1] == [batch_size, beam_size]

        assert ((0 <= top_p) & (top_p <= 5)).all()

        # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(log_probs.size()) == [batch_size, beam_size]

    @pytest.mark.parametrize("p_val", [-1.0, 1.2, 1.1, float("inf")])
    def test_p_val(self, p_val):
        with pytest.raises(ValueError):
            initial_predictions = torch.tensor([0] * 5)
            take_step = take_step_with_timestep
            beam_size = 3
            p_sampler = TopPSampler(p=p_val, with_replacement=True)

            top_k, log_probs = BeamSearch(
                self.end_index, beam_size=beam_size, max_steps=10, sampler=p_sampler
            ).search(initial_predictions, {}, take_step)

    def test_top_k_search(self):
        initial_predictions = torch.tensor([0] * 5)
        beam_size = 3
        take_step = take_step_with_timestep
        k_sampler = TopKSampler(k=5, with_replacement=True)

        top_k, log_probs = BeamSearch(
            self.end_index, beam_size=beam_size, max_steps=10, sampler=k_sampler
        ).search(initial_predictions, {}, take_step)

        beam_size = beam_size or 1
        batch_size = 5

        # top_p should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(top_k.size())[:-1] == [batch_size, beam_size]

        assert ((0 <= top_k) & (top_k <= 5)).all()

        # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(log_probs.size()) == [batch_size, beam_size]

    @pytest.mark.parametrize("k_val", [-1, 0])
    def test_k_val(self, k_val):
        with pytest.raises(ValueError):
            initial_predictions = torch.tensor([0] * 5)
            take_step = take_step_with_timestep
            beam_size = 3
            k_sampler = TopKSampler(k=k_val, with_replacement=True)

            top_k, log_probs = BeamSearch(
                self.end_index, beam_size=beam_size, max_steps=10, sampler=k_sampler
            ).search(initial_predictions, {}, take_step)

    def test_stochastic_beam_search(self):
        initial_predictions = torch.tensor([0] * 5)
        batch_size = 5
        beam_size = 3
        take_step = take_step_with_timestep

        gumbel_sampler = GumbelSampler()

        top_k, log_probs = BeamSearch(
            self.end_index, beam_size=beam_size, max_steps=10, sampler=gumbel_sampler
        ).search(initial_predictions, {}, take_step)

        # top_p should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(top_k.size())[:-1] == [batch_size, beam_size]

        assert ((0 <= top_k) & (top_k <= 5)).all()

        # log_probs should be shape `(batch_size, beam_size, max_predicted_length)`.
        assert list(log_probs.size()) == [batch_size, beam_size]

        # Check to make sure that once the end index is predicted, all subsequent tokens
        # must be the end index. This has been tested on toy examples in which
        for batch in top_k:
            for beam in batch:
                reached_end = False
                for token in beam:
                    if token == self.end_index:
                        reached_end = True
                    if reached_end:
                        assert token == self.end_index

    def test_params_sampling(self):
        beam_search = BeamSearch.from_params(
            Params(
                {
                    "sampler": {
                        "type": "top-k",
                        "k": 4,
                    },
                    "beam_size": 2,
                    "end_index": 7,
                }
            )
        )
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7
        assert beam_search.sampler is not None

    def test_params_p_sampling(self):
        beam_search = BeamSearch.from_params(
            Params(
                {
                    "sampler": {
                        "type": "top-p",
                        "p": 0.8,
                    },
                    "beam_size": 2,
                    "end_index": 7,
                }
            )
        )
        assert beam_search.beam_size == 2
        assert beam_search._end_index == 7
        assert beam_search.sampler is not None

    def test_multinomial_sampler(self):
        sampler = MultinomialSampler(temperature=0.9)

        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {"foo": "bar"})

        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)
        assert all([x < 4 for x in classes[0]])
        assert all([x > 1 for x in classes[1]])

    def test_top_k_sampler(self):
        sampler = TopKSampler(k=3, temperature=0.9)

        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {"foo": "bar"})

        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)

        assert all([x > 0 and x < 4 for x in classes[0]])
        assert all([x > 1 and x < 5 for x in classes[1]])

    def test_top_p_sampler(self):
        sampler = TopPSampler(p=0.8, temperature=0.9)

        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 3, {"foo": "bar"})

        assert probabilities.size() == classes.size()
        assert classes.size() == (2, 3)

        assert all([x > 0 and x < 4 for x in classes[0]])
        assert all([x > 1 and x < 5 for x in classes[1]])

        # Make sure the filtered classes include the first class that exceeds p
        sampler = TopPSampler(p=0.7, temperature=1.0)

        probabilities, classes, state = sampler.sample_nodes(log_probabilities, 2, {"foo": "bar"})

        assert all([x == 2 or x == 3 or x == 1 for x in classes[0]])
        assert all([x == 2 or x == 3 for x in classes[1]])

    def test_gumbel_sampler(self):
        sampler = GumbelSampler()
        num_classes = len(log_probabilities[0])
        sampler_state = sampler.init_state(log_probabilities, batch_size=2, num_classes=num_classes)

        log_probs, indices, state = sampler.sample_beams(log_probabilities, 3, sampler_state)

        assert log_probs.size() == indices.size()
        assert indices.size() == (2, 3)

        # Make sure the probabilities are sorted.
        _, sorted_indices = log_probs.sort(dim=-1, descending=True)
        assert (sorted_indices == torch.arange(3).unsqueeze(0)).all()

        assert all([x >= 0 and x < 4 for x in indices[0]])
        assert all([x > 1 and x <= 5 for x in indices[1]])
