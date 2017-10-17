from typing import List, Tuple

import torch


class DecoderState:
    def __init__(self,
                 encoder_outputs: torch.Tensor,
                 encoder_output_mask: torch.Tensor,
                 hidden_state: torch.Tensor,
                 score: torch.Tensor,
                 action_history: Tuple[int] = None) -> None:
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask
        self.hidden_state = hidden_state
        self.score = score
        self.action_history = action_history or ()

    def get_output_mask(self) -> torch.Tensor:
        return None

    def get_valid_actions(self) -> List[int]:
        return None

    def transition(self, action_log_probs: torch.Tensor, hidden_state: torch.Tensor) -> 'DecoderState':
        _, predicted_actions = torch.max(action_log_probs, 1)
        self.hidden_state = hidden_state
        self.outputs_so_far.append(predicted_actions)
        self.log_probs.append(action_log_probs)
        return self
