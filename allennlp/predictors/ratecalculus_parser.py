import os
from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

DEFAULT_EXECUTOR_JAR = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = 'data/'

@Predictor.register('ratecalculus-parser')
class RateCalculusParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.ratecalculus_semantic_parser.RateCalculusSemanticParser`
    model.
    """

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
        question_text = json_dict["question"]

        # pylint: disable=protected-access
        tokenized_question = self._dataset_reader._tokenizer.tokenize(question_text.lower())  # type: ignore
        # pylint: enable=protected-access
        instance = self._dataset_reader.text_to_instance(question_text,  # type: ignore
                                                         tokenized_question=tokenized_question)
        extra_info = {'question_tokens': tokenized_question}
        return instance, extra_info

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)
        outputs['answer'] = self._execute_logical_form(outputs['logical_form'])

        return_dict.update(outputs)
        return sanitize(return_dict)

    @staticmethod
    def _execute_logical_form(logical_form):
        """
        The parameters are written out to files which the jar file reads and then executes the
        logical form.
        """
        logical_form_filename = os.path.join(SEMPRE_DIR, 'logical_forms.txt')
        with open(logical_form_filename, 'w') as temp_file:
            temp_file.write(logical_form + '\n')

        #WARNING: CHANGME
        return "-1"
