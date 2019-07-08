from allennlp.common import Registrable
from allennlp.common.util import JsonDict

class Attacker(Registrable):
    """
    an ``Attacker`` will modify an input (e.g., add or delete tokens) to try to change an AllenNLP Predictor's output in a desired manner (e.g., make it incorrect).
    """
    def __init__(self, predictor):
        self.predictor = predictor
    def attack_from_json(self, inputs:JsonDict):
        """
        Modifies the input to change the model's prediction in some desired manner, and returns the sanitized tokens.
        """
        raise NotImplementedError("you should implement this if you want to do model attacks")