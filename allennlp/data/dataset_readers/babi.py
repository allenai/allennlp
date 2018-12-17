from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("babi")
class BAbIReader(DatasetReader):
    """
    Reads data in the bAbI tasks format as formulated in 
    Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.
    (see https://arxiv.org/abs/1502.05698)

    Parameters
    ----------
    keep_sentences: ``bool``, optional, (default = ``False``)
        Whether to keep each sentence in the context or to concatenate them.
        Default is ``False`` that corresponds to concatenation.
    subset: ``int`` , optional, (default = ``10000``)
        How many stories to retrieve from the dataset.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    """

    def __init__(self,
                 keep_sentences: bool = False,
                 subset: int = 10000,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:

        super().__init__(lazy)
        self._subset = subset
        self._keep_sentences = keep_sentences
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            dataset = dataset_file.readlines()
        
        aggregated_dataset = []
        for line in dataset:

            if '?' in line:
                q, a = line.replace('?', '').split('\t')[:-1]
                line = (None, q.split()[1:], a)
            else:
                line = line.replace('.', ' .').split()

            if line[0] == '1':
                aggregated_dataset.append([line[1:]])
            else:
                aggregated_dataset[-1].append(line[1:])

        aggregated_stories_dataset = []
        for i in range(len(aggregated_dataset)):
            story = aggregated_dataset[i]

            substories = [[]]
            for line in story:
                substories[-1].append(line)
                if isinstance(line, tuple):
                    substories.append(substories[-1][:-1])

            aggregated_stories_dataset += substories[:-1]
        
        logger.info("Reading the dataset")
        for story in aggregated_stories_dataset[:self._subset]:
            yield self.text_to_instance(story[:-1], story[-1][0], story[-1][1])

    @overrides
    def text_to_instance(self,
                         context: List[List[str]],
                         question: List[str],
                         answer: str) -> Instance:

        fields = {}

        if self._keep_sentences:
            fields['context'] = ListField([TextField([Token(word) for word in line], self._token_indexers)
                                           for line in context])
        else:
            fields['context'] = TextField([Token(word) for line in context for word in line], self._token_indexers)
            
        fields['question'] = TextField([Token(word) for word in question], self._token_indexers)
        fields['answer'] = TextField([Token(answer)], self._token_indexers)
        
        metadata = {'context': context, 'question': question, 'answer': answer}
        
        fields['metadata'] = MetadataField(metadata)
        
        return Instance(fields)
