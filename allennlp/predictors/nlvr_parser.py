import json

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('nlvr-parser')
class NlvrParserPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        if 'worlds' in json_dict:
            # This is grouped data
            worlds = json_dict['worlds']
            if isinstance(worlds, str):
                worlds = json.loads(worlds)
        else:
            structured_rep = json_dict['structured_rep']
            if isinstance(structured_rep, str):
                structured_rep = json.loads(structured_rep)
            worlds = [structured_rep]
        identifier = json_dict['identifier'] if 'identifier' in json_dict else None
        instance = self._dataset_reader.text_to_instance(sentence=sentence,  # type: ignore
                                                         structured_representations=worlds,
                                                         identifier=identifier)
        return instance

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        if "identifier" in outputs:
            # Returning CSV lines for official evaluation
            identifier = outputs["identifier"]
            denotation = outputs["denotations"][0][0]
            return f"{identifier},{denotation}\n"
        else:
            return json.dumps(outputs) + "\n"
