"""
A specification for defining task cards (derived from model cards).
Motivation: A model's capabilities and limitations are dependent on
the task definition. Thus, it is helpful to separate the information
in the model card that comes from specifically the task itself.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from allennlp.common.from_params import FromParams


@dataclass(frozen=True)
class TaskCard(FromParams):
    """
    The `TaskCard` stores information about the task. It is modeled after the
    `ModelCard`.

    # Parameters

    id : `str`
        The task id.
        Example: `"rc"` for reading comprehension.
    name : `str`, optional
        The (display) name of the task.
    description : `str`, optional
        Description of the task.
        Example: "Textual Entailment (TE) is the task of predicting whether,
                 for a pair of sentences, the facts in the first sentence necessarily
                 imply the facts in the second."
    expected_inputs : `str`, optional
        All expected inputs and their format.
        Example: (For a reading comprehension task)
                 Passage (text string), Question (text string)
    expected_outputs : `str`, optional
        All expected outputs and their format.
        Example: (For a reading comprehension task)
                 Answer span (start token position and end token position).
    examples : `Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]`, optional
        List of examples for the task. Each example dict should contain as keys the
        `expected_inputs`.
        Example: (For textual entailment)
                 [{"premise": "A handmade djembe was on display at the Smithsonian.",
                   "hypothesis": "Visitors could see the djembe."}]
    scope_and_limitations: `str`, optional
        This discusses the scope of the task based on how it is defined, and any limitations.
        Example: "The Textual Entailment task is in some sense "NLP-complete", and you
                  should not expect any current model to cover every possible aspect of
                  entailment. Instead, you should think about what the model was trained
                  on to see whether it could reasonably capture the phenomena that you
                  are querying it with."
    """

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    expected_inputs: Optional[str] = None
    expected_outputs: Optional[str] = None
    scope_and_limitations: Optional[str] = None
    examples: Optional[Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]] = None
