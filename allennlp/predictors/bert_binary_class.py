from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import numpy as np

@Predictor.register('bert_binary_class')
class MCQABinPredictor(Predictor):
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        if 'id' in json_dict:
            item_id = json_dict['id']
        else:
            item_id = ''
        question = json_dict['question']['stem']

        # Our model was usually trained on 5 answers, so we will need to filter out 2 answers..
        choice_list = [choice['text'] for choice in json_dict['question']['choices']]
        instance = self._dataset_reader.text_to_instance(item_id, question, choice_list)
        predictions = self.predict_instance(instance)
        predicted_answer = int(predictions['label_probs'] > 0.5)

        example = {'id':item_id,'pred_answer':predicted_answer,'question': question, 'predictions': predictions['label_probs'], \
                   'predictions_logits': predictions['label_logits'],'choice_list':choice_list,'answerKey':json_dict['answerKey']}

        return example
