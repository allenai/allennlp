from typing import cast, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers.semantic_parsing.quarel import QuarelDatasetReader
from allennlp.predictors.predictor import Predictor
from allennlp.semparse.contexts.quarel_utils import get_explanation, from_qr_spec_string
from allennlp.semparse.contexts.quarel_utils import words_from_entity_string, from_entity_cues_string


@Predictor.register('quarel-parser')
class QuarelParserPredictor(Predictor):
    """
    Wrapper for the quarel_semantic_parser model.
    """
    def _my_json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        """

        # Make a cast here to satisfy mypy
        dataset_reader = cast(QuarelDatasetReader, self._dataset_reader)

        # TODO: Fix protected access usage
        question_data = dataset_reader.preprocess(json_dict, predict=True)[0]

        qr_spec_override = None
        dynamic_entities = None
        if 'entitycues' in json_dict:
            entity_cues = from_entity_cues_string(json_dict['entitycues'])
            dynamic_entities = dataset_reader._dynamic_entities.copy()  # pylint: disable=protected-access
            for entity, cues in entity_cues.items():
                key = "a:" + entity
                entity_strings = [words_from_entity_string(entity).lower()]
                entity_strings += cues
                dynamic_entities[key] = " ".join(entity_strings)

        if 'qrspec' in json_dict:
            qr_spec_override = from_qr_spec_string(json_dict['qrspec'])
            old_entities = dynamic_entities
            if old_entities is None:
                old_entities = dataset_reader._dynamic_entities.copy()  # pylint: disable=protected-access
            dynamic_entities = {}
            for qset in qr_spec_override:
                for entity in qset:
                    key = "a:" + entity
                    value = old_entities.get(key, words_from_entity_string(entity).lower())
                    dynamic_entities[key] = value

        question = question_data['question']
        tokenized_question = dataset_reader._tokenizer.tokenize(question.lower())  # pylint: disable=protected-access
        world_extractions = question_data.get('world_extractions')

        instance = dataset_reader.text_to_instance(question,
                                                   world_extractions=world_extractions,
                                                   qr_spec_override=qr_spec_override,
                                                   dynamic_entities_override=dynamic_entities)

        world_extractions_out = {"world1": "N/A", "world2": "N/A"}
        if world_extractions is not None:
            world_extractions_out.update(world_extractions)

        extra_info = {'question': json_dict['question'],
                      'question_tokens': tokenized_question,
                      "world_extractions": world_extractions_out}
        return instance, extra_info

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance, _ = self._my_json_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._my_json_to_instance(inputs)
        world = instance.fields['world'].metadata  # type: ignore
        outputs = self._model.forward_on_instance(instance)

        answer_index = outputs['answer_index']
        if answer_index == 0:
            answer = "A"
        elif answer_index == 1:
            answer = "B"
        else:
            answer = "None"
        outputs['answer'] = answer

        return_dict.update(outputs)

        if answer != "None":
            explanation = get_explanation(return_dict['logical_form'],
                                          return_dict['world_extractions'],
                                          answer_index,
                                          world)
        else:
            explanation = [{"header": "No consistent interpretation found!", "content": []}]

        return_dict['explanation'] = explanation
        return sanitize(return_dict)
