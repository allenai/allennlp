from overrides import overrides
from allennlp.common.util import JsonDict 
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, LabelField
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides
from typing import List, Dict 
import numpy as np
from copy import deepcopy

@Predictor.register('sentence-tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)
    
    @overrides
    def predictions_to_labeled_instances(self, instance: Instance, outputs: Dict[str, np
        .ndarray]) -> List[Instance]:
        """
        Imagine an NER model predicts three named entities (each one with potentially multiple tokens). For each individual entity, we create a new Instance that has the label set to only that entity and the rest of the tokens are labeled as outside. We then return a list of those Instances.

        For example:
        Mary  went to Seattle to visit Microsoft Research
        U-Per  O    O   U-Loc  O   O     B-Org     L-Org
    
        We create three instances. 
        Mary  went to Seattle to visit Microsoft Research
        U-Per  O    O    O     O   O       O         O
        
        Mary  went to Seattle to visit Microsoft Research
        O      O    O   U-LOC  O   O       O         O
        
        Mary  went to Seattle to visit Microsoft Research
        O      O    O    O     O   O     B-Org     L-Org
        """
        tags = outputs['tags']
        tag_list = []

        i = 0
        while (i < len(tags)): 
            tag = tags[i]
            # if its a U, add it to the list
            if tag[0] == 'U':
                cur_tags = [t if idx == i else 'O' for idx, t in enumerate(tags)]
                tag_list.append(cur_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == 'B':
                begin_idx = i            
                while (tag[0] != 'L'):
                    i += 1
                    tag = tags[i]
                end_idx = i
                cur_tags = [t if idx >= begin_idx and idx <= end_idx else 'O' for idx, t in enumerate(tags)]
                tag_list.append(cur_tags)
            i += 1

        # Creates a new instance for each contiguous tag
        instance_list = []
        for tags in tag_list:
            new_instance = deepcopy(instance)
            new_instance.add_field('tags', ListField([LabelField(tag) for tag in tags]), self._model.vocab)
            instance_list.append(new_instance)

        return instance_list 