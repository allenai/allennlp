from typing import List

from allennlp.nn.beam_search import BeamSearch

class InteractiveBeamSearch(BeamSearch):
    """
    This is designed to be a drop-in replacement for ``BeamSearch``
    that allows you to force the beam search down a certain path.
    You can use this to create model demos that allow the user
    to interact with the beam search part of the model.
    """
    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None,
                 initial_sequence: List[int] = None) -> None:
        super().__init__(end_index, max_steps, beam_size, per_node_beam_size)
        self._initial_sequence = initial_sequence or []
