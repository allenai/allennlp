from typing import Dict

import torch

class ChecklistState:
    """
    This class keeps track of checklist related variables that are used while training a coverage
    based semantic parser (or any other kind of transition based constrained decoder). This is
    inteded to be used within a ``DecoderState``.

    Parameters
    ----------
    terminal_actions : ``torch.Tensor``
        A vector containing the indices of terminal actions, required for computing checklists for
        next states based on current actions. The idea is that we will build checklists
        corresponding to the presence or absence of just the terminal actions. But in principle,
        they can be all actions that are relevant to checklist computation.
    checklist_target : ``torch.Tensor``
        Targets corresponding to checklist that indicate the states in which we want the checklist to
        ideally be. It is the same size as ``terminal_actions``, and it contains 1 for each corresponding
        action in the list that we want to see in the final logical form, and 0 for each corresponding
        action that we do not.
    checklist_mask : ``torch.Tensor``
        Mask corresponding to ``terminal_actions``, indicating which of those actions are relevant
        for checklist computation. For example, if the parser is penalizing non-agenda terminal
        actions, all the terminal actions are relevant.
    checklist : ``torch.Tensor``
        A checklist indicating how many times each action in its agenda has been chosen previously.
        It contains the actual counts of the agenda actions.
    """
    def __init__(self,
                 terminal_actions: torch.Tensor,
                 checklist_target: torch.Tensor,
                 checklist_mask: torch.Tensor,
                 checklist: torch.Tensor) -> None:
        self.terminal_actions = terminal_actions
        self.checklist_target = checklist_target
        self.checklist_mask = checklist_mask
        self.checklist = checklist
        # Mapping from batch action indices to indices in any of the four vectors above.
        self.terminal_indices_dict: Dict[int, int] = {}
        for checklist_index, batch_action_index in enumerate(terminal_actions.data.cpu()):
            action_index = int(batch_action_index[0])
            if action_index == -1:
                continue
            self.terminal_indices_dict[action_index] = checklist_index

    def update(self, action: torch.Tensor) -> 'ChecklistState':
        """
        Takes an action index, updates checklist and returns an updated state.
        """
        checklist_addition = (self.terminal_actions == action).float()
        new_checklist = self.checklist + checklist_addition
        new_checklist_state = ChecklistState(terminal_actions=self.terminal_actions,
                                             checklist_target=self.checklist_target,
                                             checklist_mask=self.checklist_mask,
                                             checklist=new_checklist)
        return new_checklist_state

    def get_balance(self) -> torch.Tensor:
        return self.checklist_mask * (self.checklist_target - self.checklist)
