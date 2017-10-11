from typing import List

import torch


class DecoderState:
    def __init__(self,
                 encoder_outputs: torch.Tensor,
                 encoder_output_mask: torch.Tensor,
                 hidden_state: torch.Tensor,
                 outputs_so_far: List[torch.Tensor] = None,
                 log_probs: List[torch.Tensor] = None) -> None:
        self.encoder_outputs = encoder_outputs
        self.encoder_output_mask = encoder_output_mask
        self.hidden_state = hidden_state
        self.outputs_so_far = outputs_so_far or []
        self.log_probs = log_probs or []

    def get_output_mask(self) -> torch.Tensor:
        return None

    def get_valid_actions(self) -> List[int]:
        return None

    def update(self, action_log_probs: torch.Tensor, hidden_state: torch.Tensor) -> 'DecoderState':
        _, predicted_actions = torch.max(action_log_probs, 1)
        self.hidden_state = hidden_state
        self.outputs_so_far.append(predicted_actions)
        self.log_probs.append(action_log_probs)
        return self
