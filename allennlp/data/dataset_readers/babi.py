import logging

from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)


@DatasetReader.register("babi")
class BabiReader(DatasetReader):
    """
    Reads one single task in the bAbI tasks format as formulated in
    Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks
    (https://arxiv.org/abs/1502.05698). Since this class handle a single file,
    if one wants to load multiple tasks together it has to merge them into a
    single file and use this reader.

    Registered as a `DatasetReader` with name "babi".

    # Parameters

    keep_sentences : `bool`, optional, (default = `False`)
        Whether to keep each sentence in the context or to concatenate them.
        Default is `False` that corresponds to concatenation.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        keep_sentences: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self._keep_sentences = keep_sentences
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)

        with open(file_path) as dataset_file:
            dataset = dataset_file.readlines()

        logger.info("Reading the dataset")

        context: List[List[str]] = [[]]
        for line in dataset:
            if "?" in line:
                question_str, answer, supports_str = line.replace("?", " ?").split("\t")
                question = question_str.split()[1:]
                supports = [int(support) - 1 for support in supports_str.split()]

                yield self.text_to_instance(context, question, answer, supports)
            else:
                new_entry = line.replace(".", " .").split()[1:]

                if line[0] == "1":
                    context = [new_entry]
                else:
                    context.append(new_entry)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        context: List[List[str]],
        question: List[str],
        answer: str,
        supports: List[int],
    ) -> Instance:

        fields: Dict[str, Field] = {}

        if self._keep_sentences:
            context_field_ks = ListField(
                [
                    TextField([Token(word) for word in line], self._token_indexers)
                    for line in context
                ]
            )

            fields["supports"] = ListField(
                [IndexField(support, context_field_ks) for support in supports]
            )
        else:
            context_field = TextField(
                [Token(word) for line in context for word in line], self._token_indexers
            )

        fields["context"] = context_field_ks if self._keep_sentences else context_field
        fields["question"] = TextField([Token(word) for word in question], self._token_indexers)
        fields["answer"] = TextField([Token(answer)], self._token_indexers)

        return Instance(fields)
