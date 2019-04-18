from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bert_mc_qa")
class BertMCQAReader(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:
        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 pretrained_model: str,
                 instance_per_choice: bool = False,
                 max_pieces: int = 512,
                 num_choices: int = 5,
                 answer_only: bool = False,
                 syntax: str = "arc",
                 restrict_num_choices: int = None,
                 skip_id_regex: str = None,
                 ignore_context: bool = False,
                 context_syntax: str = "c#q#a",
                 sample: int = -1) -> None:
        super().__init__()
        #self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        lower_case = not '-cased' in pretrained_model
        self._word_splitter = BertBasicWordSplitter(do_lower_case=lower_case)
        self._max_pieces = max_pieces
        self._instance_per_choice = instance_per_choice
        self._sample = sample
        self._num_choices = num_choices
        self._syntax = syntax
        self._context_syntax = context_syntax
        self._answer_only = answer_only
        self._restrict_num_choices = restrict_num_choices
        self._skip_id_regex = skip_id_regex
        self._ignore_context = ignore_context


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())

                if debug > 0:
                    logger.info(item_json)

                if self._syntax == 'quarel' or self._syntax == 'quarel_preamble':
                    item_json = self._normalize_mc(item_json)
                    if debug > 0:
                        logger.info(item_json)
                elif self._syntax == 'vcr':
                    item_json = self._normalize_vcr(item_json)
                    if debug > 0:
                        logger.info(item_json)

                item_id = item_json["id"]
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue
                context = item_json.get("para")
                if self._ignore_context:
                    context = None
                question_text = item_json["question"]["stem"]

                if self._syntax == 'quarel_preamble':
                    context, question_text = question_text.split(". ", 1)

                if self._answer_only:
                    question_text = ""

                choice_label_to_id = {}
                choice_text_list = []
                choice_context_list = []

                any_correct = False
                choice_id_correction = 0

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):

                    if self._restrict_num_choices and len(choice_text_list) == self._restrict_num_choices:
                        if not any_correct:
                            choice_text_list.pop(-1)
                            choice_context_list.pop(-1)
                            choice_id_correction += 1
                        else:
                            break

                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id - choice_id_correction

                    choice_text = choice_item["text"]
                    choice_context = choice_item.get("para")
                    if self._ignore_context:
                        choice_context = None

                    choice_text_list.append(choice_text)
                    choice_context_list.append(choice_context)

                    is_correct = 0
                    if item_json.get('answerKey') == choice_label:
                        is_correct = 1
                        if any_correct:
                            raise ValueError("More than one correct answer found for {item_json}!")
                        any_correct = True

                    if self._restrict_num_choices \
                            and len(choice_text_list) == self._restrict_num_choices \
                            and not any_correct:
                        continue


                if not any_correct and 'answerKey' in item_json:
                    raise ValueError("No correct answer found for {item_json}!")

                answer_id = choice_label_to_id[item_json["answerKey"]]
                # Pad choices with empty strings if not right number
                if len(choice_text_list) != self._num_choices:
                    choice_text_list = (choice_text_list + self._num_choices * [''])[:self._num_choices]
                    choice_context_list = (choice_context_list + self._num_choices * [None])[:self._num_choices]
                    if answer_id >= self._num_choices:
                        logging.warning(f"Skipping question with more than {self._num_choices} answers: {item_json}")
                        continue

                yield self.text_to_instance(
                    item_id,
                    question_text,
                    choice_text_list,
                    answer_id,
                    context,
                    choice_context_list,
                    debug)


    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int = None,
                         context: str = None,
                         choice_context_list: List[str] = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        qa_fields = []
        segment_ids_fields = []
        qa_tokens_list = []
        for idx, choice in enumerate(choice_list):
            choice_context = context
            if choice_context_list is not None and choice_context_list[idx] is not None:
                choice_context = choice_context_list[idx]
            qa_tokens, segment_ids = self.bert_features_from_qa(question, choice, choice_context)
            qa_field = TextField(qa_tokens, self._token_indexers)
            segment_ids_field = SequenceLabelField(segment_ids, qa_field)
            qa_fields.append(qa_field)
            qa_tokens_list.append(qa_tokens)
            segment_ids_fields.append(segment_ids_field)
            if debug > 0:
                logger.info(f"qa_tokens = {qa_tokens}")
                logger.info(f"segment_ids = {segment_ids}")

        fields['question'] = ListField(qa_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            "correct_answer_index": answer_id,
            "question_tokens_list": qa_tokens_list
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        if debug > 0:
            logger.info(f"answer_id = {answer_id}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, max_length):
        """
        Truncate a from the start and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def _normalize_mc(self, json: JsonDict) -> JsonDict:
        split = self.split_mc_question(json['question'])
        if split is None:
            raise ValueError("No question split found for {json}!")
            return None
        answer_index = json['answer_index']
        res = {"id": json['id'],
               'question': split,
               'answerKey': split['choices'][answer_index]['label']}
        return res

    def _normalize_vcr(self, json: JsonDict) -> JsonDict:
        unisex_names = ["Avery", "Riley", "Jordan", "Angel", "Parker", "Sawyer", "Peyton",
                        "Quinn", "Blake", "Hayden", "Taylor", "Alexis", "Rowan"]
        obj = json['objects']
        qa = [json['question']] + json['answer_choices']
        qa_updated = []
        for tokens in qa:
            qa_new = []
            for token in tokens:
                if isinstance(token, str):
                    qa_new.append(token)
                else:
                    entities = []
                    for ref in token:
                        entity = obj[ref]
                        if entity == 'person':
                            entity = unisex_names[ref % len(unisex_names)]
                        entities.append(entity)
                    qa_new.append(" and ".join(entities))
            qa_updated.append(" ".join(qa_new))
        answer_index = json['answer_label']
        question = qa_updated[0]
        choices = [{'text': answer, 'label': str(idx)} for idx, answer in enumerate(qa_updated[1:])]
        return {"id": json['annot_id'],
                "question": {"stem": question, "choices": choices},
                "answerKey": str(answer_index)
                }

    @staticmethod
    def split_mc_question(question, min_choices=2):
        choice_sets = [["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"],
                       ["1", "2", "3", "4", "5"],
                       ["G", "H", "J", "K"],
                       ['a', 'b', 'c', 'd', 'e']]
        patterns = [r'\(#\)', r'#\)', r'#\.']
        for pattern in patterns:
            for choice_set in choice_sets:
                regex = pattern.replace("#","(["+"".join(choice_set)+"])")
                labels = [m.group(1) for m in re.finditer(regex, question)]
                if len(labels) >= min_choices and labels == choice_set[:len(labels)]:
                    splits = [s.strip() for s in re.split(regex, question)]
                    return {"stem": splits[0],
                            "choices": [{"text": splits[i+1],
                                         "label": splits[i]} for i in range(1, len(splits)-1, 2)]}
        return None

    def bert_features_from_qa(self, question: str, answer: str, context: str = None):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        question_tokens = self._word_splitter.split_words(question)
        if context is not None:
            context_tokens = self._word_splitter.split_words(context)
            question_tokens = context_tokens + [sep_token] + question_tokens
        choice_tokens = self._word_splitter.split_words(answer)
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, self._max_pieces - 3)

        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        # AllenNLP already add [CLS]
        #tokens = question_tokens + [sep_token] + choice_tokens
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(1, len(choice_tokens) + 1))
        return tokens, segment_ids