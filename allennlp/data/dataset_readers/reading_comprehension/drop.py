import itertools
import json
import logging
import string
from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any

from overrides import overrides
from word2number.w2n import word_to_num

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, MetadataField, LabelField, ListField, \
    SequenceLabelField, SpanField, IndexField
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.reading_comprehension.util import (IGNORED_TOKENS,
                                                                      STRIPPED_CHARACTERS,
                                                                      make_reading_comprehension_instance,
                                                                      split_tokens_by_hyphen)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer


logger = logging.getLogger(__name__)


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}


@DatasetReader.register("drop")
class DropReader(DatasetReader):
    """
    Reads a JSON-formatted DROP dataset file and returns instances in a few different possible
    formats.  The input format is complicated; see the test fixture for an example of what it looks
    like.  The output formats all contain a question ``TextField``, a passage ``TextField``, and
    some kind of answer representation.  Because DROP has instances with several different kinds of
    answers, this dataset reader allows you to filter out questions that do not have answers of a
    particular type (e.g., remove questions that have numbers as answers, if you model can only
    give passage spans as answers).  We typically return all possible ways of arriving at a given
    answer string, and expect models to marginalize over these possibilities.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    passage_length_limit : ``int``, optional (default=None)
        If specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        If specified, we will cut the question if the length of passage exceeds this limit.
    skip_when_all_empty: ``List[str]``, optional (default=None)
        In some cases such as preparing for training examples, you may want to skip some examples
        when there are no gold labels. You can specify on what condition should the examples be
        skipped. Currently, you can put "passage_span", "question_span", "addition_subtraction",
        or "counting" in this list, to tell the reader skip when there are no such label found.
        If not specified, we will keep all the examples.
    instance_format: ``str``, optional (default="drop")
        We try to be generous in providing a few different formats for the instances in DROP,
        in terms of the ``Fields`` that we return for each ``Instance``, to allow for several
        different kinds of models.  "drop" format will do processing to detect numbers and
        various ways those numbers can be arrived at from the passage, and return ``Fields``
        related to that.  "bert" format only allows passage spans as answers, and provides a
        "question_and_passage" field with the two pieces of text joined as BERT expects.
        "squad" format provides the same fields that our BiDAF and other SQuAD models expect.
    relaxed_span_match_for_finding_labels : ``bool``, optional (default=True)
        DROP dataset contains multi-span answers, and the date-type answers are usually hard to
        find exact span matches for, also.  In order to use as many examples as possible
        to train the model, we may not want a strict match for such cases when finding the gold
        span labels. If this argument is true, we will treat every span in the multi-span
        answers as correct, and every token in the date answer as correct, too.  Because models
        trained on DROP typically marginalize over all possible answer positions, this is just
        being a little more generous in what is being marginalized.  Note that this will not
        affect evaluation.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None,
                 skip_when_all_empty: List[str] = None,
                 instance_format: str = "drop",
                 relaxed_span_match_for_finding_labels: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction", "counting"], \
                f"Unsupported skip type: {item}"
        self.instance_format = instance_format
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        logger.info("Reading the dataset")
        kept_count, skip_count = 0, 0
        for passage_id, passage_info in dataset.items():
            passage_text = passage_info["passage"]
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
            for question_answer in passage_info["qa_pairs"]:
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 passage_tokens)
                if instance is not None:
                    kept_count += 1
                    yield instance
                else:
                    skip_count += 1
        logger.info(f"Skipped {skip_count} questions, kept {kept_count} questions.")

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         passage_tokens: List[Token] = None) -> Union[Instance, None]:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
            passage_tokens = split_tokens_by_hyphen(passage_tokens)
        question_tokens = self._tokenizer.tokenize(question_text)
        question_tokens = split_tokens_by_hyphen(question_tokens)
        if self.passage_length_limit is not None:
            passage_tokens = passage_tokens[: self.passage_length_limit]
        if self.question_length_limit is not None:
            question_tokens = question_tokens[: self.question_length_limit]

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            # Currently we only use the first annotated answer here, but actually this doesn't affect
            # the training, because we only have one annotation for the train set.
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts = []
        for answer_text in answer_texts:
            answer_tokens = self._tokenizer.tokenize(answer_text)
            answer_tokens = split_tokens_by_hyphen(answer_tokens)
            tokenized_answer_texts.append(' '.join(token.text for token in answer_tokens))

        if self.instance_format == "squad":
            valid_passage_spans = \
                self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            if not valid_passage_spans:
                if "passage_span" in self.skip_when_all_empty:
                    return None
                else:
                    valid_passage_spans.append((len(passage_tokens) - 1, len(passage_tokens) - 1))
            return make_reading_comprehension_instance(question_tokens,
                                                       passage_tokens,
                                                       self._token_indexers,
                                                       passage_text,
                                                       valid_passage_spans,
                                                       # this `answer_texts` will not be used for evaluation
                                                       answer_texts,
                                                       additional_metadata={
                                                               "original_passage": passage_text,
                                                               "original_question": question_text,
                                                               "passage_id": passage_id,
                                                               "question_id": question_id,
                                                               "valid_passage_spans": valid_passage_spans,
                                                               "answer_annotations": answer_annotations})
        elif self.instance_format == "bert":
            question_concat_passage_tokens = question_tokens + [Token("[SEP]")] + passage_tokens
            valid_passage_spans = []
            for span in self.find_valid_spans(passage_tokens, tokenized_answer_texts):
                # This span is for `question + [SEP] + passage`.
                valid_passage_spans.append((span[0] + len(question_tokens) + 1,
                                            span[1] + len(question_tokens) + 1))
            if not valid_passage_spans:
                if "passage_span" in self.skip_when_all_empty:
                    return None
                else:
                    valid_passage_spans.append((len(question_concat_passage_tokens) - 1,
                                                len(question_concat_passage_tokens) - 1))
            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans}
            return self.make_bert_drop_instance(question_tokens,
                                                passage_tokens,
                                                question_concat_passage_tokens,
                                                self._token_indexers,
                                                passage_text,
                                                answer_info,
                                                additional_metadata={
                                                        "original_passage": passage_text,
                                                        "original_question": question_text,
                                                        "passage_id": passage_id,
                                                        "question_id": question_id,
                                                        "answer_annotations": answer_annotations})
        elif self.instance_format == "drop":
            numbers_in_passage = []
            number_indices = []
            for token_index, token in enumerate(passage_tokens):
                number = self.convert_word_to_number(token.text)
                if number is not None:
                    numbers_in_passage.append(number)
                    number_indices.append(token_index)
            # hack to guarantee minimal length of padded number
            numbers_in_passage.append(0)
            number_indices.append(-1)
            numbers_as_tokens = [Token(str(number)) for number in numbers_in_passage]

            valid_passage_spans = \
                self.find_valid_spans(passage_tokens, tokenized_answer_texts) if tokenized_answer_texts else []
            valid_question_spans = \
                self.find_valid_spans(question_tokens, tokenized_answer_texts) if tokenized_answer_texts else []

            target_numbers = []
            # `answer_texts` is a list of valid answers.
            for answer_text in answer_texts:
                number = self.convert_word_to_number(answer_text)
                if number is not None:
                    target_numbers.append(number)
            valid_signs_for_add_sub_expressions: List[List[int]] = []
            valid_counts: List[int] = []
            if answer_type in ["number", "date"]:
                valid_signs_for_add_sub_expressions = self.find_valid_add_sub_expressions(numbers_in_passage,
                                                                                          target_numbers)
            if answer_type in ["number"]:
                # Currently we only support count number 0 ~ 9
                numbers_for_count = list(range(10))
                valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

            type_to_answer_map = {"passage_span": valid_passage_spans,
                                  "question_span": valid_question_spans,
                                  "addition_subtraction": valid_signs_for_add_sub_expressions,
                                  "counting": valid_counts}

            if self.skip_when_all_empty \
                    and not any(type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
                return None

            answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
                           "counts": valid_counts}

            return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    numbers_as_tokens,
                                                    number_indices,
                                                    self._token_indexers,
                                                    passage_text,
                                                    answer_info,
                                                    additional_metadata={
                                                            "original_passage": passage_text,
                                                            "original_question": question_text,
                                                            "original_numbers": numbers_in_passage,
                                                            "passage_id": passage_id,
                                                            "question_id": question_id,
                                                            "answer_info": answer_info,
                                                            "answer_annotations": answer_annotations})
        else:
            raise ValueError(f"Expect the instance format to be \"drop\", \"squad\" or \"bert\", "
                             f"but got {self.instance_format}")

    @staticmethod
    def make_marginal_drop_instance(question_tokens: List[Token],
                                    passage_tokens: List[Token],
                                    number_tokens: List[Token],
                                    number_indices: List[int],
                                    token_indexers: Dict[str, TokenIndexer],
                                    passage_text: str,
                                    answer_info: Dict[str, Any] = None,
                                    additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        question_offsets = [(token.idx, token.idx + len(token.text)) for token in question_tokens]

        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        question_field = TextField(question_tokens, token_indexers)
        fields["passage"] = passage_field
        fields["question"] = question_field
        number_index_fields: List[Field] = [IndexField(index, passage_field) for index in number_indices]
        fields["number_indices"] = ListField(number_index_fields)
        # This field is actually not required in the model,
        # it is used to create the `answer_as_plus_minus_combinations` field, which is a `SequenceLabelField`.
        # We cannot use `number_indices` field for creating that, because the `ListField` will not be empty
        # when we want to create a new empty field. That will lead to error.
        numbers_in_passage_field = TextField(number_tokens, token_indexers)
        metadata = {"original_passage": passage_text,
                    "passage_token_offsets": passage_offsets,
                    "question_token_offsets": question_offsets,
                    "question_tokens": [token.text for token in question_tokens],
                    "passage_tokens": [token.text for token in passage_tokens],
                    "number_tokens": [token.text for token in number_tokens],
                    "number_indices": number_indices}
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]

            passage_span_fields: List[Field] = \
                [SpanField(span[0], span[1], passage_field) for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

            question_span_fields: List[Field] = \
                [SpanField(span[0], span[1], question_field) for span in answer_info["answer_question_spans"]]
            if not question_span_fields:
                question_span_fields.append(SpanField(-1, -1, question_field))
            fields["answer_as_question_spans"] = ListField(question_span_fields)

            add_sub_signs_field: List[Field] = []
            for signs_for_one_add_sub_expression in answer_info["signs_for_add_sub_expressions"]:
                add_sub_signs_field.append(SequenceLabelField(signs_for_one_add_sub_expression,
                                                              numbers_in_passage_field))
            if not add_sub_signs_field:
                add_sub_signs_field.append(SequenceLabelField([0] * len(number_tokens),
                                                              numbers_in_passage_field))
            fields["answer_as_add_sub_expressions"] = ListField(add_sub_signs_field)

            count_fields: List[Field] = [LabelField(count_label, skip_indexing=True)
                                         for count_label in answer_info["counts"]]
            if not count_fields:
                count_fields.append(LabelField(-1, skip_indexing=True))
            fields["answer_as_counts"] = ListField(count_fields)

        metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def make_bert_drop_instance(question_tokens: List[Token],
                                passage_tokens: List[Token],
                                question_concat_passage_tokens: List[Token],
                                token_indexers: Dict[str, TokenIndexer],
                                passage_text: str,
                                answer_info: Dict[str, Any] = None,
                                additional_metadata: Dict[str, Any] = None) -> Instance:
        additional_metadata = additional_metadata or {}
        fields: Dict[str, Field] = {}
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        # This is separate so we can reference it later with a known type.
        passage_field = TextField(passage_tokens, token_indexers)
        question_field = TextField(question_tokens, token_indexers)
        fields["passage"] = passage_field
        fields["question"] = question_field
        question_and_passage_field = TextField(question_concat_passage_tokens, token_indexers)
        fields['question_and_passage'] = question_and_passage_field

        metadata = {'original_passage': passage_text, 'passage_token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens], }

        if answer_info:
            metadata['answer_texts'] = answer_info["answer_texts"]

            passage_span_fields: List[Field] = \
                [SpanField(span[0], span[1], question_and_passage_field)
                 for span in answer_info["answer_passage_spans"]]
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, question_and_passage_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key]
                           for key in ["month", "day", "year"] if key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def convert_word_to_number(word: str, try_to_include_more_numbers=False):
        """
        Currently we only support limited types of conversion.
        """
        if try_to_include_more_numbers:
            # strip all punctuations from the sides of the word, except for the negative sign
            punctruations = string.punctuation.replace('-', '')
            word = word.strip(punctruations)
            # some words may contain the comma as deliminator
            word = word.replace(",", "")
            # word2num will convert hundred, thousand ... to number, but we skip it.
            if word in ["hundred", "thousand", "million", "billion", "trillion"]:
                return None
            try:
                number = word_to_num(word)
            except ValueError:
                try:
                    number = int(word)
                except ValueError:
                    try:
                        number = float(word)
                    except ValueError:
                        number = None
            return number
        else:
            no_comma_word = word.replace(",", "")
            if no_comma_word in WORD_NUMBER_MAP:
                number = WORD_NUMBER_MAP[no_comma_word]
            else:
                try:
                    number = int(no_comma_word)
                except ValueError:
                    number = None
            return number

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token],
                         answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List[int],
                                       targets: List[int],
                                       max_number_of_numbers_to_consider: int = 2) -> List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    if eval_value in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices
