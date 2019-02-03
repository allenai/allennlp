import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# ALON - for line profiler
try:
    profile
except NameError:
    profile = lambda x: x

@DatasetReader.register("multiqa_bert")
class BERTQAReader(DatasetReader):
    """
    Reads a JSON-formatted Quesiton Answering in Context (QuAC) data file
    and returns a ``Dataset`` where the ``Instances`` have four fields: ``question``, a ``ListField``,
    ``passage``, another ``TextField``, and ``span_start`` and ``span_end``, both ``ListField`` composed of
    IndexFields`` into the ``passage`` ``TextField``.
    Two ``ListField``, composed of ``LabelField``, ``yesno_list`` and  ``followup_list`` is added.
    We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_text_lists'] and ``metadata['token_offsets']``.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 sample_size: int = -1) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sample_size = sample_size
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def build_instances(self, header, instances):

        # bucketing by QuestionID
        instance_list = instances
        instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
        intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                                  range(len(split_inds) - 1)]

        # sorting
        sorting_keys = ['question_tokens', 'tokens']
        instances_with_lengths = []
        for instance in per_question_instances:
            padding_lengths = {key: len(instance[0][key]) for key in sorting_keys}
            instance_with_lengths = ([padding_lengths[field_name] for field_name in sorting_keys], instance)
            instances_with_lengths.append(instance_with_lengths)
        #instances_with_lengths.sort(key=lambda x: x[0])
        # random shuffle training
        random.seed(8)
        random.shuffle(instances_with_lengths)

        per_question_instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

        # selecting instaces to add
        filtered_instances = []
        for question_instances in per_question_instances:
            if header['split_type'] == 'dev':
                instances_to_add = question_instances
            else:
                # choose at most 2 instances from the same question:
                if len(question_instances) > 2:
                    # This part is inspired by Clark and Gardner, 17 - oversample the highest ranking documents.
                    # In thier work they use only instances with answers, so we will find the highest
                    # ranking instance with an answer (this also insures we have at least one answer in the chosen instances)
                    inst_with_answers = [inst for inst in question_instances if inst['answers'] != []]
                    instances_to_add = random.sample(inst_with_answers[0:2], 1)
                    # we assume each question will be visited once in an epoch
                    question_instances.remove(instances_to_add[0])
                    instances_to_add += random.sample(question_instances, 1)

                else:
                    instances_to_add = question_instances

                # Require at least one answer:
                if not any(inst['answers'] != [] for inst in instances_to_add):
                    continue

            filtered_instances += instances_to_add

        #logger.info("multiqa+: yielding %d instances ", len(filtered_instances))
        for inst_num, inst in enumerate(filtered_instances):
            # for bert changing the order of the passage token only for now...
            # very patchy but it's just for testing

            char_offset = 6 # accounting for the new [CLS] + space
            new_passage_tokens = [('[CLS]',0)]
            new_passage = '[CLS] '
            for t in inst['question_tokens']:
                new_passage_tokens.append((t[0],t[1] + char_offset))
            new_passage += inst['metadata']['question'] + ' '
            char_offset = len(new_passage) # question length + space

            # we also need to account for change in answer location
            token_offset = len(inst['question_tokens']) + 1

            # finding ' ^SEP^ ' and replacing it

            char_offset_to_sep = None
            for ind,t in enumerate(inst['tokens']):
                if t[0] == ' ^SEP^ ':
                    split_point = ind
                    char_offset_to_sep = inst['tokens'][split_point + 1][1]
                    token_offset = len(inst['question_tokens']) + 1 - split_point
                if char_offset_to_sep is not None:
                    if ind + token_offset >= 400:
                        x=1
                        break
                    t[1] = t[1] - char_offset_to_sep + char_offset + len('[SEP] ')
            if ind != len(inst['tokens']) - 1:
                new_passage_tokens += [('[SEP]', char_offset)] + inst['tokens'][split_point + 1:ind]
            else:
                new_passage_tokens += [('[SEP]', char_offset)] + inst['tokens'][split_point + 1:]



            # creating the new passage with [SEP]
            new_passage += "[SEP] " + inst['text'][char_offset_to_sep:] + ' '

            new_answers = []
            for answer in inst['answers']:
                if answer[1] + token_offset < len(new_passage_tokens):
                    new_answers.append([answer[0] + token_offset, answer[1] + token_offset, answer[2]])
                else:
                    print('lost one answer!')



            new_passage_tokens.append(("[SEP]",len(new_passage)))
            new_passage += "[SEP]"

            # sanity check:
            if False:
                for answer in new_answers:
                    char_idx_start = new_passage_tokens[answer[0]][1]
                    if answer[1] + 1 >= len(new_passage_tokens):
                        char_idx_end = len(new_passage)
                    else:
                        char_idx_end = new_passage_tokens[answer[1] + 1][1]

                    if re.match(r'\b{0}\b'.format(re.escape(answer[2])), \
                        new_passage[char_idx_start:char_idx_end], re.IGNORECASE) is None:
                        if (answer[2].lower().strip() != \
                                new_passage[char_idx_start:char_idx_end].lower().strip()):
                            x = 1


            tokenized_paragraph = [Token(text=t[0], idx=t[1]) for t in new_passage_tokens]
            question_tokens = [Token(text=t[0], idx=t[1]) for t in inst['question_tokens']]
            instance = util.make_reading_comprehension_instance_multiqa_multidoc(question_tokens,
                                                                                 tokenized_paragraph,
                                                                                 self._token_indexers,
                                                                                 new_passage,
                                                                                 new_answers,
                                                                                 inst['metadata'],
                                                                                 header)

            yield instance

    @profile
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading the dataset")

        # supporting multi dataset training:
        instances = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            # if `file_path` is a URL, redirect to the cache
            logger.info("Reading file at %s", single_file_path)

            if single_file_path.find('jsonl') > 0:
                single_file_path_cached = cached_path(single_file_path)
                with zipfile.ZipFile(single_file_path_cached, 'r') as myzip:
                    with myzip.open(myzip.namelist()[0]) as myfile:
                        header = json.loads(myfile.readline())['header']
                        number_of_yielded_instances = 0
                        for line,example in enumerate(myfile):
                            # header
                            instances.append(json.loads(example))

                            # making sure not to take all instances of the same question
                            if len(instances)>10000 and instances[-1]['metadata']['question_id'] \
                                                    != instances[-2]['metadata']['question_id']:
                                remainder = instances[-1]
                                instances = instances[:-1]
                                for instance in self.build_instances(header, instances):
                                    number_of_yielded_instances += 1
                                    yield instance
                                instances = [remainder]

                            if self._sample_size != -1 and number_of_yielded_instances > self._sample_size:
                                break


                        # yielding the remainder
                        for instance in self.build_instances(header, instances):
                            yield instance



    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         paragraph: List[str],
                         span_starts: List[List[int]] = None,
                         span_ends: List[List[int]] = None,
                         tokenized_paragraph: List[List[Token]] = None,
                         additional_metadata: Dict[str, Any] = None) -> Instance:
        # pylint: disable=arguments-differ
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.

        tokenized_paragraph = tokenized_paragraph or []

        # Building answer_token_span_list shape: [answer_type, paragraph, questions , answer list]
        # Span_starts_list is a list of dim [answer types, question num] each values is (paragraph num, answer start char offset)
        answer_token_span_list = {'answers':[],'distractor_answers':[]}
        for answer_type in ['answers', 'distractor_answers']:
            passage_offsets = [(token.idx, token.idx + len(token.text)) for token in tokenized_paragraph]

            token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in zip(span_starts[answer_type], span_ends[answer_type]):
                (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                             (char_span_start, char_span_end))
                if error:
                    logger.debug("Passage: %s", paragraph)
                    logger.debug("Passage tokens: %s", tokenized_paragraph)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug("Tokens in answer: %s", tokenized_paragraph[span_start:span_end + 1])
                    logger.debug("Answer: %s", paragraph[char_span_start:char_span_end])
                token_spans.append((span_start, span_end))

            answer_token_span_list[answer_type].append(token_spans)

        question_tokens = self._tokenizer.tokenize(question_text)

        return util.make_reading_comprehension_instance_multiqa_multidoc(question_tokens,
                                                             tokenized_paragraph,
                                                             self._token_indexers,
                                                             paragraph,
                                                             answer_token_span_list,
                                                             additional_metadata)
