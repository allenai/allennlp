from inspect import signature
from typing import List, Callable, Tuple, Dict, cast, TypeVar, Optional
import warnings

import torch

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.nn.samplers.sampler import Sampler
from allennlp.nn.samplers.samplers import TopKSampler
from allennlp.nn.samplers.samplers import TopPSampler
from allennlp.nn.samplers.samplers import GumbelMaxSampler


StateType = Dict[str, torch.Tensor]
StepFunctionTypeWithTimestep = Callable[
    [torch.Tensor, StateType, int], Tuple[torch.Tensor, StateType]
]
StepFunctionTypeNoTimestep = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]

StepFunctionType = TypeVar(
    "StepFunctionType", StepFunctionTypeWithTimestep, StepFunctionTypeNoTimestep
)
"""
The type of step function that can be passed to [`BeamSearch.search`](#search).

This can either be [`StepFunctionTypeWithTimestep`](#stepfunctiontypewithtimestep)
or [`StepFunctionTypeNoTimestep`](#stepfunctiontypenotimestep).
"""


class BeamSearch(Registrable):
    """
    Implements the beam search algorithm for decoding the most likely sequences.

    # Parameters

    end_index : `int`
        The index of the "stop" or "end" token in the target vocabulary.
    max_steps : `int`, optional (default = `50`)
        The maximum number of decoding steps to take, i.e. the maximum length
        of the predicted sequences.
    beam_size : `int`, optional (default = `10`)
        The width of the beam used.
    per_node_beam_size : `int`, optional (default = `beam_size`)
        The maximum number of candidates to consider per node, at each step in the search.
        If not given, this just defaults to `beam_size`. Setting this parameter
        to a number smaller than `beam_size` may give better results, as it can introduce
        more diversity into the search. See
        [*Beam Search Strategies for Neural Machine Translation*, Freitag and Al-Onaizan, 2017]
        (https://arxiv.org/abs/1702.01806).
    sampler : `Sampler`, optional (default = `None`)
        A sampler that can be used to select subsequent tokens.
        If not given, search defaults to selecting the top `per_node_beam_size` tokens at each
        step.
    """

    default_implementation = "without_sampling"

    def __init__(
        self,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: int = None,
        sampler: Sampler = None,
    ) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.sampler = sampler

    @staticmethod
    def _reconstruct_sequences(predictions, backpointers):
        # Reconstruct the sequences.
        # shape: [(batch_size, beam_size, 1)]
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]

        # shape: (batch_size, beam_size)
        cur_backpointers = backpointers[-1]

        for timestep in range(len(predictions) - 2, 0, -1):
            # shape: (batch_size, beam_size, 1)
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)

            reconstructed_predictions.append(cur_preds)

            # shape: (batch_size, beam_size)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)

        # shape: (batch_size, beam_size, 1)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)

        reconstructed_predictions.append(final_preds)

        return reconstructed_predictions

    @torch.no_grad()
    def search(
        self,
        start_predictions: torch.Tensor,
        start_state: StateType,
        step: StepFunctionType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a starting state and a step function, apply beam search to find the
        most likely target sequences.

        # Notes

        If your step function returns `-inf` for some log probabilities
        (like if you're using a masked log-softmax) then some of the "best"
        sequences returned may also have `-inf` log probability. Specifically
        this happens when the beam size is smaller than the number of actions
        with finite log probability (non-zero probability) returned by the step function.
        Therefore if you're using a mask you may want to check the results from `search`
        and potentially discard sequences with non-finite log probability.

        # Parameters

        start_predictions : `torch.Tensor`
            A tensor containing the initial predictions with shape `(batch_size,)`.
            Usually the initial predictions are just the index of the "start" token
            in the target vocabulary.

        start_state : `StateType`
            The initial state passed to the `step` function. Each value of the state dict
            should be a tensor of shape `(batch_size, *)`, where `*` means any other
            number of dimensions.

        step : `StepFunctionType`
            A function that is responsible for computing the next most likely tokens,
            given the current state and the predictions from the last time step.
            The function should accept two or three arguments:

            - a tensor of shape `(group_size,)` representing the index of the predicted
            tokens from the last time step,
            - the current state, a `StateType`, and
            - optionally, the timestep, an `int`.

            The `group_size` will be `batch_size * beam_size`, except in the initial
            step, for which it will just be `batch_size`.

            The function is expected to return a tuple, where the first element
            is a tensor of shape `(group_size, target_vocab_size)` containing
            the log probabilities of the tokens for the next step, and the second
            element is the updated state. The tensor in the state should have shape
            `(group_size, *)`, where `*` means any other number of dimensions.

        # Returns

        `Tuple[torch.Tensor, torch.Tensor]`
            Tuple of `(predictions, log_probabilities)`, where `predictions`
            has shape `(batch_size, beam_size, max_steps)` and `log_probabilities`
            has shape `(batch_size, beam_size)`.
        """
        step_signature = signature(step)
        if len(step_signature.parameters) < 3:
            # If the step function we're given does not take the time step argument, wrap it
            # in one that does.
            old_step = cast(StepFunctionTypeNoTimestep, step)

            def new_step(
                last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], time_step: int
            ):
                return old_step(last_predictions, state)

            return self._search(start_predictions, start_state, new_step)
        else:
            return self._search(
                start_predictions, start_state, cast(StepFunctionTypeWithTimestep, step)
            )

    def _search(
        self,
        start_predictions: torch.Tensor,
        start_state: StateType,
        step: StepFunctionTypeWithTimestep,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = start_predictions.size()[0]

        # List of (batch_size, beam_size) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions: List[torch.Tensor] = []

        # List of (batch_size, beam_size) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from.
        backpointers: List[torch.Tensor] = []

        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_log_probabilities, state = step(start_predictions, start_state, 0)

        num_classes = start_class_log_probabilities.size()[1]

        # Make sure `per_node_beam_size` is not larger than `num_classes`.
        if self.per_node_beam_size > num_classes:
            raise ConfigurationError(
                f"Target vocab size ({num_classes:d}) too small "
                f"relative to per_node_beam_size ({self.per_node_beam_size:d}).\n"
                f"Please decrease beam_size or per_node_beam_size."
            )

        # Get the initial predicted classed and their log probabilities.
        # If this search includes sampling, select the tokens using the designated sampler.
        # Else, select the top `beam_size` tokens.
        # shape: (batch_size, beam_size), (batch_size, beam_size)
        if self.sampler is not None:
            start_top_log_probabilities, start_predicted_classes = self.sampler(
                start_class_log_probabilities, start_class_log_probabilities, num_samples=self.beam_size
            )
        else:
            (
                start_top_log_probabilities,
                start_predicted_classes,
            ) = start_class_log_probabilities.topk(self.beam_size)

        # Some samplers add perturb the log probabilities with random noise, such
        # as the GumbelMaxSampler, so we need to recover and keep track of the
        # true log probabilities here.
        last_true_log_probabilities: Optional[torch.Tensor] = None
        if self.sampler is not None:
            last_true_log_probabilities = torch.gather(
                start_class_log_probabilities, 1, start_predicted_classes
            )

        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            warnings.warn(
                "Empty sequences predicted. You may want to increase the beam size or ensure "
                "your step function is working properly.",
                RuntimeWarning,
            )
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities

        # The log probabilities for the last time step.
        # shape: (batch_size, beam_size)
        last_log_probabilities = start_top_log_probabilities

        # shape: [(batch_size, beam_size)]
        predictions.append(start_predicted_classes)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            if state_tensor is None:
                continue
            multilayer_rnn_decoder = state_tensor.dim() == 3 and key in {
                "decoder_hidden",
                "decoder_context",
            }

            if multilayer_rnn_decoder:
                # shape: (num_layers, batch_size * beam_size, *)
                num_layers, _, *last_dims = state_tensor.size()
                state[key] = (
                    state_tensor.unsqueeze(2)
                    .expand(num_layers, batch_size, self.beam_size, *last_dims)
                    .reshape(num_layers, batch_size * self.beam_size, *last_dims)
                )
            else:
                # shape: (batch_size * beam_size, *)
                _, *last_dims = state_tensor.size()
                state[key] = (
                    state_tensor.unsqueeze(1)
                    .expand(batch_size, self.beam_size, *last_dims)
                    .reshape(batch_size * self.beam_size, *last_dims)
                )

        for timestep in range(self.max_steps - 1):
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._end_index`,
            # then we can stop early.
            if (last_predictions == self._end_index).all():
                break
            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            # shape: (batch_size * beam_size, num_classes)
            class_log_probabilities, state = step(last_predictions, state, timestep + 1)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # shape: (batch_size * beam_size, 1)
            last_log_probabilities = last_log_probabilities.unsqueeze(-1).reshape(
                batch_size * self.beam_size, 1
            )

            if last_true_log_probabilities is not None:
                # shape: (batch_size * beam_size, 1)
                last_true_log_probabilities = last_true_log_probabilities.unsqueeze(-1).reshape(
                    batch_size * self.beam_size, 1
                )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities,
            )

            summed_log_probabilities = torch.add(last_log_probabilities, cleaned_log_probabilities)

            if last_true_log_probabilities is not None:
                # add the last true probs to previous sequence probs
                summed_true_log_probabilities = torch.add(last_true_log_probabilities, cleaned_log_probabilities)
                

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            if self.sampler is not None:
                top_log_probabilities, predicted_classes = self.sampler(
                    summed_true_log_probabilities, summed_log_probabilities, num_samples=self.per_node_beam_size
                )
            else:
                top_log_probabilities, predicted_classes = summed_log_probabilities.topk(
                    self.per_node_beam_size
                )

            summed_top_log_probabilities = top_log_probabilities
            
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_log_probabilities.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # Track the true probabilities in addition to perturbed probabilities
            # for use in samplers that may perturb probabilities
            if last_true_log_probabilities is not None:
                # shape: (batch_size * beam_size, per_node_beam_size)
                expanded_true_log_probabilities = last_true_log_probabilities.expand(
                    batch_size * self.beam_size, self.per_node_beam_size
                )

                # shape: (batch_size * beam_size, per_node_beam_size)
                summed_true_log_probabilities = (
                    cleaned_log_probabilities.gather(1, predicted_classes)
                    + expanded_true_log_probabilities
                )

                # shape: (batch_size, beam_size * per_node_beam_size)
                reshaped_summed_true_probabilities = summed_true_log_probabilities.reshape(
                    batch_size, self.beam_size * self.per_node_beam_size
                )

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )

            # Use the beam indices to extract the corresponding classes.
            # shape: (batch_size, beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(
                1, restricted_beam_indices
            )

            # track the true probabilities for the selected beams
            # shape: (batch_size, beam_size)
            if last_true_log_probabilities is not None:
                restricted_true_log_probabilities = reshaped_summed_true_probabilities.gather(
                    1, restricted_beam_indices
                )

                last_true_log_probabilities = restricted_true_log_probabilities

            predictions.append(restricted_predicted_classes)

            # shape: (batch_size, beam_size)
            last_log_probabilities = restricted_beam_log_probs

            # The beam indices come from a `beam_size * per_node_beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by per_node_beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            # shape: (batch_size, beam_size)
            backpointer = restricted_beam_indices // self.per_node_beam_size
            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                if state_tensor is None:
                    continue
                multilayer_rnn_decoder = state_tensor.dim() == 3 and key in {
                    "decoder_hidden",
                    "decoder_context",
                }

                if multilayer_rnn_decoder:
                    # shape: (num_layers, batch_size * beam_size, *)
                    num_layers, _, *last_dims = state_tensor.size()
                    expanded_backpointer = backpointer.view(
                        batch_size, self.beam_size, *([1] * len(last_dims))
                    ).expand(batch_size, self.beam_size, *last_dims)
                    expanded_backpointer = expanded_backpointer.unsqueeze(0).repeat(
                        num_layers, 1, 1, 1
                    )
                    # shape: (num_layers, batch_size * beam_size, *)
                    state[key] = (
                        state_tensor.reshape(num_layers, batch_size, self.beam_size, *last_dims)
                        .gather(2, expanded_backpointer)
                        .reshape(num_layers, batch_size * self.beam_size, *last_dims)
                    )
                else:
                    _, *last_dims = state_tensor.size()
                    # shape: (batch_size, beam_size, *)
                    expanded_backpointer = backpointer.view(
                        batch_size, self.beam_size, *([1] * len(last_dims))
                    ).expand(batch_size, self.beam_size, *last_dims)
                    # shape: (batch_size * beam_size, *)
                    state[key] = (
                        state_tensor.reshape(batch_size, self.beam_size, *last_dims)
                        .gather(1, expanded_backpointer)
                        .reshape(batch_size * self.beam_size, *last_dims)
                    )

        if not torch.isfinite(last_log_probabilities).all():
            warnings.warn(
                "Infinite log probabilities encountered. Some final sequences may not make sense. "
                "This can happen when the beam size is larger than the number of valid (non-zero "
                "probability) transitions that the step function produces.",
                RuntimeWarning,
            )

        reconstructed_predictions = self._reconstruct_sequences(predictions, backpointers)

        # shape: (batch_size, beam_size, max_steps)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        # If a sampler is used, return the true probabilities instead of perturbed probabilities
        if last_true_log_probabilities is not None:
            return all_predictions, last_true_log_probabilities

        return all_predictions, last_log_probabilities

    @classmethod
    def without_sampling(
        cls,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: int = None,
    ) -> "BeamSearch":
        """
        Given an index of the end token in target vocabulary, return a `BeamSearch` object
        that can be used to find `beam_size` candidate sequences.
        """
        return cls(
            end_index=end_index,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
        )

    @classmethod
    def top_k_sampling(
        cls,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        k: int = 1,
        temperature: float = 1.0,
    ) -> "BeamSearch":
        """
        Given an index of the end token in target vocabulary, return a `BeamSearch` object
        that can be used to find `beam_size` candidate sequences, found by sampling from
        the top `k` tokens to choose from at each step.
        """
        # Make sure `k` is a valid threshold
        if type(k) is not int or k < 1:
            raise ConfigurationError(
                f'{"value of selection threshold `k` invalid."}'
                f'{"`k` must be a positive `int`."}'
            )
        sampler_k = TopKSampler(k, temperature)
        return cls(
            end_index=end_index,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=1,
            sampler=sampler_k,
        )

    @classmethod
    def top_p_sampling(
        cls,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        p: float = 0.9,
        temperature: float = 1.0,
    ) -> "BeamSearch":
        """
        Given an index of the end token in target vocabulary, return a `BeamSearch` object
        that can be used to find `beam_size` candidate sequences, found by sampling from
        tokens which cumulatively make up the top `p` probability of the total tokens to
        choose from.
        """
        # Make sure `p` is a valid cumulative probability threshold.
        if type(p) is not float or p < 0.0 or p > 1.0:
            raise ConfigurationError(
                f'{"value of cumulative probability threshold `p`=({p}) too small."}'
                f'{"`p` must be a float between `0.0` and `1.0`"}'
            )

        sampler_p = TopPSampler(p, temperature)
        return cls(
            end_index=end_index,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=1,
            sampler=sampler_p,
        )

    @classmethod
    def stochastic_beam_search(
        cls,
        end_index: int,
        max_steps: int = 50,
        beam_size: int = 10,
        per_node_beam_size: int = None,
    ) -> "BeamSearch":
        """
        Given an index of the end token in target vocabulary, return a `BeamSearch` object
        that can be used to find `beam_size` candidate sequences, found by using Stochastic
        Beam Search, which leverages Gumbel-TopK sampling. See
        [*Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
        Sequences Without Replacement*, W Kool, H Van Hoof and M Welling, 2010]
        (https://arxiv.org/abs/1903.06059).
        """

        gumbel_sampler = GumbelMaxSampler()
        return cls(
            end_index=end_index,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size or beam_size,
            sampler=gumbel_sampler,
        )


BeamSearch.register("without_sampling", constructor="without_sampling")(BeamSearch)
BeamSearch.register("top_p_sampling", constructor="top_p_sampling")(BeamSearch)
BeamSearch.register("top_k_sampling", constructor="top_k_sampling")(BeamSearch)
BeamSearch.register("stochastic_beam_search", constructor="stochastic_beam_search")(BeamSearch)
