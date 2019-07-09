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
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def attack_from_json(self,
                         inputs: JsonDict,
                         name_of_input_field_to_attack: str = 'tokens',
                         name_of_grad_input_field: str = 'grad_input_1',
                         ignore_tokens: List[str] = ["@@NULL@@"]) -> JsonDict: # pylint: disable=dangerous-default-value
        """
        This function modifies the input to change the model's prediction in some desired manner
        (e.g., an adversarial attack).

        The input is the same as what goes to a `Predictor`. The output is a JsonDict
        containing the sanitized final tokens. For example, the final tokens might be a slight
        modification to the input tokens that cause the model to change its prediction.

        `name_of_input_field_to_attack` is for example `tokens`, it says what the input
        field is called. `name_of_grad_input_field` is for example `grad_input_1`, which
        is a key into a grads dictionary.
        """
        raise NotImplementedError("you should implement this if you want to do model attacks")
