from typing import List, Callable, Tuple, Dict

import torch


StateType = Dict[str, torch.Tensor]  # pylint: disable=invalid-name
StepFunctionType = Callable[[torch.Tensor, StateType], Tuple[torch.Tensor, StateType]]  # pylint: disable=invalid-name


class BeamSearch:

    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10) -> None:
        self._end_index = end_index
        self.beam_size = beam_size
        self.max_steps = max_steps

    def search(self,
               start_predictions: torch.Tensor,
               start_state: StateType,
               step: StepFunctionType) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use beam search decoding to find the highest probability sequences.

        Returns
        -------
        ``Tuple[torch.Tensor, torch.Tensor]``
            ``(predictions, log_probabilities)``, where ``predictions``
            has shape ``(batch_size, beam_size, max_steps)`` and ``log_probabilities``
            has shape ``(batch_size, beam_size)``.
        """
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
        start_class_log_probabilities, state = step(start_predictions, start_state)
        # shape: (batch_size, num_classes)

        start_top_log_probabilities, start_predicted_classes = \
                start_class_log_probabilities.topk(self.beam_size)
        # shape: (batch_size, beam_size), (batch_size, beam_size)

        num_classes = start_class_log_probabilities.size()[1]

        # The log probabilities for the last time step.
        last_log_probabilities = start_top_log_probabilities
        # shape: (batch_size, beam_size)

        predictions.append(start_predicted_classes)
        # shape: [(batch_size, beam_size)]

        # Log probability tensor that mandates that the end token is selected.
        log_probs_after_end = start_class_log_probabilities.new_full(
                (batch_size * self.beam_size, num_classes),
                float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.
        # shape: (batch_size * beam_size, num_classes)

        # Set the same state for each element in the beam.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.\
                    unsqueeze(1).\
                    expand(batch_size, self.beam_size, *last_dims).\
                    reshape(batch_size * self.beam_size, *last_dims)
            # shape: (batch_size * beam_size, *)

        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)
            # shape: (batch_size * beam_size,)

            # Take a step. This get the predicted log probs of the next classes
            # and updates the state.
            class_log_probabilities, state = step(last_predictions, state)
            # shape: (batch_size * beam_size, num_classes)

            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                    batch_size * self.beam_size,
                    num_classes
            )
            # shape: (batch_size * beam_size, num_classes)

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            cleaned_log_probabilities = torch.where(
                    last_predictions_expanded == self._end_index,
                    log_probs_after_end,
                    class_log_probabilities
            )
            # shape: (batch_size * beam_size, num_classes)

            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(self.beam_size)
            # shape: (batch_size * beam_size, beam_size), (batch_size * beam_size, beam_size)

            # Here we expand the last log probabilities to (batch_size * beam_size, beam_size)
            # so that we can add them to the current log probs for this timestep.
            # This lets us maintain the log probability of each element on the beam.
            expanded_last_log_probabilities = last_log_probabilities.\
                    unsqueeze(2).\
                    expand(batch_size, self.beam_size, self.beam_size).\
                    reshape(batch_size * self.beam_size, self.beam_size)
            # shape: (batch_size * beam_size, beam_size)

            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities
            # shape: (batch_size * beam_size, beam_size)

            reshaped_summed = summed_top_log_probabilities.\
                    reshape(batch_size, self.beam_size * self.beam_size)
            # shape: (batch_size, beam_size * beam_size)

            reshaped_predicted_classes = predicted_classes.\
                    reshape(batch_size, self.beam_size * self.beam_size)
            # shape: (batch_size, beam_size * beam_size)

            # Keep only the top `beam_size` beam indices.
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)
            # shape: (batch_size, beam_size), (batch_size, beam_size)

            # Use the beam indices to extract the corresponding classes.
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)
            # shape: (batch_size, beam_size)

            predictions.append(restricted_predicted_classes)

            last_log_probabilities = restricted_beam_log_probs
            # shape: (batch_size, beam_size)

            # The beam indices come from a `beam_size * beam_size` dimension where the
            # indices with a common ancestor are grouped together. Hence
            # dividing by beam_size gives the ancestor. (Note that this is integer
            # division as the tensor is a LongTensor.)
            backpointer = restricted_beam_indices / self.beam_size
            # shape: (batch_size, beam_size)

            backpointers.append(backpointer)

            # Keep only the pieces of the state tensors corresponding to the
            # ancestors created this iteration.
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.\
                        unsqueeze(2).\
                        expand(batch_size, self.beam_size, *last_dims)
                # shape: (batch_size, beam_size, *)

                state[key] = state_tensor.\
                        reshape(batch_size, self.beam_size, *last_dims).\
                        gather(1, expanded_backpointer).\
                        reshape(batch_size * self.beam_size, *last_dims)
                # shape: (batch_size * beam_size, *)

        assert len(predictions) == self.max_steps,\
               "len(predictions) not equal to num_decoding_steps"
        assert len(backpointers) == self.max_steps - 1,\
               "len(backpointers) not equal to num_decoding_steps"

        # Reconstruct the sequences.
        reconstructed_predictions = [predictions[self.max_steps - 1].unsqueeze(2)]
        # shape: [(batch_size, beam_size, 1)]

        cur_backpointers = backpointers[self.max_steps - 2]
        # shape: (batch_size, beam_size)

        for timestep in range(self.max_steps - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            # shape: (batch_size, beam_size, 1)

            reconstructed_predictions.append(cur_preds)

            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
            # shape: (batch_size, beam_size)

        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        # shape: (batch_size, beam_size, 1)

        reconstructed_predictions.append(final_preds)

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        # shape: (batch_size, beam_size, max_steps)

        return all_predictions, last_log_probabilities
