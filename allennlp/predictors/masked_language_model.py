from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('masked_lm_predictor')
class MaskedLanguageModelPredictor(Predictor):    
    def predict(self, tokens: str, mask_positions: List[int], target_ids: List[int]) -> JsonDict:
        return self.predict_json({"tokens" : tokens, 
                                  "mask_positions": mask_positions, 
                                  "target_ids": target_ids})        
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"tokens": "..."}``.
        """
        tokens = json_dict["tokens"]
        mask_positions = json_dict["mask_positions"]
        target_ids = json_dict["targets_ids"]
        return self._dataset_reader.text_to_instance(tokens=tokens, mask_positions=mask_positions, target_ids=target_ids)
