from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from typing import List, Dict
import numpy as np
from allennlp.data.fields import LabelField
from allennlp.data.dataset_readers import StanfordSentimentTreeBankDatasetReader
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

@Predictor.register('text_classifier')
class TextClassifierPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model
    """
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        sentence = json_dict["sentence"]            
        if isinstance(self._dataset_reader, StanfordSentimentTreeBankDatasetReader):        
            tokenizer = WordTokenizer() 
            sentence = [str(t) for t in tokenizer.tokenize(sentence)]                        
        return self._dataset_reader.text_to_instance(sentence)

    @overrides        
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np.ndarray]) -> List[Instance]:        
        label = np.argmax(outputs['probs'])
        instance.add_field('label', LabelField(int(label), skip_indexing=True))                    
        return [instance]