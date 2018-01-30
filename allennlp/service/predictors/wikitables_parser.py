from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('wikitables-parser')
class WikiTablesParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
        print("Got json:", json_dict)
        question_text = json_dict["question"]
        table_text = json_dict["table"]
        cells = []
        for row_index, line in enumerate(table_text.split('\n')):
            line = line.rstrip('\n')
            if row_index == 0:
                columns = line.split('\t')
            else:
                cells.append(line.split('\t'))
        table_json = {"columns": columns, "cells": cells}
        return self._dataset_reader.text_to_instance(question_text, table_json), {}
