from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('vae')
class VAEPredictor(Predictor):
    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        This is used for unconditional text generation

        ``num_to_generate`` is overloading the batch element in the model's input
        This is the correct implementation but may cause memory problems when ``num_to_generate`` is too big.
        """
        assert list(inputs.keys()) == ['num_to_generate']
        num_to_generate = int(inputs['num_to_generate'])
        response = self._model.generate(num_to_generate)
        response = sanitize(response)

        return response

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        pass
