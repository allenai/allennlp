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

def sample_contexts(instance_list,sample_size):
    random.seed(2)

    instance_list = sorted(instance_list, key=lambda x: x['metadata']['question_id'])
    intances_question_id = [instance['metadata']['question_id'] for instance in instance_list]
    split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
    per_question_instances = [instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                              range(len(split_inds) - 1)]

    random.shuffle(per_question_instances)

    sampled_contexts = []
    num_of_qas = 0
    for question_instances in per_question_instances:
        if num_of_qas > sample_size:
            break
        sampled_contexts += question_instances
        num_of_qas += 1
    return sampled_contexts

@DatasetReader.register("multiqa+combine")
class MultiQAReader(DatasetReader):
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
                 dev_sample_size: int = -1,
                 train_sample_size: int = -1,) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._dev_sample_size = dev_sample_size
        self._train_sample_size = train_sample_size
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @profile
    @overrides
    def _read(self, file_path: str):
        logger.info("Reading the dataset")

        # supporting multi dataset training:
        mixed_instance_list = []
        for ind, single_file_path in enumerate(file_path.split(',')):
            instance_list = []
            # if `file_path` is a URL, redirect to the cache
            logger.info("Reading file at %s", single_file_path)

            single_file_path_cached = cached_path(single_file_path)
            with zipfile.ZipFile(single_file_path_cached, 'r') as myzip:
                with myzip.open(myzip.namelist()[0]) as myfile:
                    header = json.loads(myfile.readline())['header']
                    for line,example in enumerate(myfile):
                        instance_list.append(json.loads(example))

            # per dataset sampling
            if header['split_type'] == 'dev' and self._dev_sample_size > -1:
                instance_list = sample_contexts(instance_list, self._dev_sample_size)
            elif header['split_type'] == 'train' and self._train_sample_size > -1:
                instance_list = sample_contexts(instance_list, self._train_sample_size)

            mixed_instance_list += instance_list


        # bucketing by QuestionID
        mixed_instance_list = sorted(mixed_instance_list, key=lambda x: x['metadata']['question_id'])
        intances_question_id = [instance['metadata']['question_id'] for instance in mixed_instance_list]
        split_inds = [0] + list(np.cumsum(np.unique(intances_question_id, return_counts=True)[1]))
        per_question_instances = [mixed_instance_list[split_inds[ind]:split_inds[ind + 1]] for ind in
                                  range(len(split_inds) - 1)]

        # sorting
        sorting_keys = ['question_tokens','tokens']
        instances_with_lengths = []
        for instance in per_question_instances:
            padding_lengths = {key: len(instance[0][key]) for key in sorting_keys}
            instance_with_lengths = ([padding_lengths[field_name] for field_name in sorting_keys], instance)
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[0])
        per_question_instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

        # selecting instaces to add
        instances = []
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

            instances += instances_to_add


        logger.info("multiqa+: yielding %d instances ", len(instances))
        for inst_num,inst in enumerate(instances):
            if inst_num % 99 == 0:
                logger.info("yeilding inst_num %d",inst_num)
            tokenized_paragraph = [Token(text=t[0], idx=t[1]) for t in inst['tokens']]
            question_tokens = [Token(text=t[0], idx=t[1]) for t in inst['question_tokens']]
            instance = util.make_reading_comprehension_instance_multiqa(question_tokens,
                                                     tokenized_paragraph,
                                                     self._token_indexers,
                                                     inst['text'],
                                                     inst['answers'],
                                                     inst['metadata'],
                                                     header)

            yield instance


