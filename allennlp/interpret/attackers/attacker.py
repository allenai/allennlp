from typing import List
from allennlp.common import Registrable
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict

class Attacker(Registrable):
    """
    an ``Attacker`` will modify an input (e.g., add or delete tokens)
    to try to change an AllenNLP Predictor's output in a desired
    manner (e.g., make it incorrect).
    """
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def attack_from_json(self,
                         inputs: JsonDict,
                         input_field_to_attack: str,
                         grad_input_field: str,
                         ignore_tokens: List[str]) -> JsonDict: # pylint: disable=dangerous-default-value
        """
        This function modifies the input to change the model's prediction in some desired manner
        (e.g., an adversarial attack).

        Parameters
        ----------
        inputs: JsonDict
            The input you want to attack, similar to the argument to a Predictor, e.g., predict_json().
        input_field_to_attack: str
            The key in the inputs JsonDict you want to attack, e.g., `tokens`.
        grad_input_field: str
            The field in the gradients dictionary that contains the input gradients.
            For example, `grad_input_1` will be the field for single input tasks. See
            get_gradients() in `Predictor` for more information on field names.
        Returns
        -------
        JsonDict
            Contains the final, sanitized input after adversarial modification.
        """
        raise NotImplementedError()
