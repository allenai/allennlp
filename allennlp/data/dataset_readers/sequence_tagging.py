from typing import List

from overrides import overrides

from . import DatasetReader
from .. import Dataset
from .. import Instance
from ...common import Params
from ..fields import TextField, TagField
from ..token_indexers import TokenIndexer, SingleIdTokenIndexer


class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging.

    Parameters
    ----------
    filename : ``str``
    token_indexers : ``List[TokenIndexer]``, optional (default=``[SingleIdTokenIndexer()]``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    """
    def __init__(self,
                 filename: str,
                 token_indexers: List[TokenIndexer] = None):
        self._filename = filename
        self._token_indexers = token_indexers or [SingleIdTokenIndexer()]

    @overrides
    def read(self):
        with open(self._filename, "r") as data_file:

            instances = []
            for line in data_file:
                tokens_and_tags = [pair.split("###") for pair in line.strip("\n").split("\t")]
                tokens = [x[0] for x in tokens_and_tags]
                tags = [x[1] for x in tokens_and_tags]

                sequence = TextField(tokens, self._token_indexers)
                sequence_tags = TagField(tags, sequence)
                instances.append(Instance({'input_tokens': sequence,
                                           'output_tags': sequence_tags}))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        filename : ``str``
        tokenizer : ``Params``, optional
        token_indexers: ``List[Params]``, optional
        """
        filename = params.pop('filename')
        token_indexers = [TokenIndexer.from_params(p)
                          for p in params.pop('token_indexers', [Params({})])]
        params.assert_empty(cls.__name__)
        return SequenceTaggingDatasetReader(filename=filename,
                                            token_indexers=token_indexers)
