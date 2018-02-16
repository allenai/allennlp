from typing import Tuple
from overrides import overrides
import os
from subprocess import run

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


DEFAULT_EXECUTOR_JAR = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = 'sempre-data/'

@Predictor.register('wikitables-parser')
class WikiTablesParserPredictor(Predictor):
    """
    Wrapper for the
    :class:`~allennlp.models.encoder_decoders.wikitables_semantic_parser.WikiTablesSemanticParser`
    model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # Load auxiliary sempre files during startup for faster logical form execution.
        os.makedirs(SEMPRE_DIR, exist_ok=True)
        if not os.path.exists(SEMPRE_DIR + 'abbreviations.tsv'):
            run(f'wget {ABBREVIATIONS_FILE}', shell=True)
            run(f'mv wikitables-abbreviations.tsv {SEMPRE_DIR}abbreviations.tsv', shell=True)
        if not os.path.exists(SEMPRE_DIR + 'grow.grammar'):
            run(f'wget {GROW_FILE}', shell=True)
            run(f'mv wikitables-grow.grammar {SEMPRE_DIR}grow.grammar', shell=True)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"question": "...", "table": "..."}``.
        """
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
        # pylint: disable=protected-access
        tokenized_question = self._dataset_reader._tokenizer.tokenize(question_text)  # type: ignore
        # pylint: enable=protected-access
        instance = self._dataset_reader.text_to_instance(question_text,  # type: ignore
                                                         table_json,
                                                         tokenized_question=tokenized_question)
        extra_info = {'question_tokens': tokenized_question}
        return instance, extra_info

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance, return_dict = self._json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance, cuda_device)

        logical_form_filename = os.path.join(SEMPRE_DIR, 'logical_forms.txt')
        with open(logical_form_filename, 'w') as f:
            f.write(outputs['logical_form'] + '\n')

        table_dir = SEMPRE_DIR + 'csv/'
        os.makedirs(table_dir, exist_ok=True)
        table_filename = 'context.csv'
        with open(table_dir + table_filename, 'w', encoding='utf-8') as f:
            f.write(inputs["table"])

        test_record = ('(example (id %s) (utterance %s) (context (graph tables.TableKnowledgeGraph %s))'
                       '(targetValue (list (description "6"))))' % ('nt-0', inputs['question'],
                                                                     table_filename))
        test_data_filename = SEMPRE_DIR + 'data.examples'
        with open(test_data_filename, 'w') as f:
            f.write(test_record)

        command = ' '.join(['java',
                            '-jar',
                            cached_path(DEFAULT_EXECUTOR_JAR),
                            test_data_filename,
                            logical_form_filename,
                            table_dir,
                            ])
        run(command, shell=True)

        denotations_file = SEMPRE_DIR + 'logical_forms_denotations.tsv'
        with open(denotations_file) as f:
            line = f.readline().split('\t')
            outputs['answer'] = line[1] if len(line) > 1 else line[0]

        return_dict.update(outputs)
        return sanitize(return_dict)
