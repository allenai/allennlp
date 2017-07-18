from typing import Dict

from overrides import overrides

from allennlp.common import Params
from allennlp.data import Dataset, DatasetReader, Instance, TokenIndexer
from allennlp.data.fields import TextField, TagField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.experiments import Registry


@Registry.register_dataset_reader("sequence_tagging")
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def read(self, file_path):
        with open(file_path, "r") as data_file:

            instances = []
            for line in data_file:
                tokens_and_tags = [pair.split("###") for pair in line.strip("\n").split("\t")]
                tokens = [x[0] for x in tokens_and_tags]
                tags = [x[1] for x in tokens_and_tags]

                sequence = TextField(tokens, self._token_indexers)
                sequence_tags = TagField(tags, sequence)
                instances.append(Instance({'tokens': sequence,
                                           'tags': sequence_tags}))
        return Dataset(instances)

    @classmethod
    def from_params(cls, params: Params):
        """
        Parameters
        ----------
        token_indexers: ``Dict[Params]``, optional
        """
        token_indexers = {}
        token_indexer_params = params.pop('token_indexers', Params({}))
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)
        # The default parameters are contained within the class,
        # so if no parameters are given we must pass None.
        if token_indexers == {}:
            token_indexers = None
        params.assert_empty(cls.__name__)
        return SequenceTaggingDatasetReader(token_indexers=token_indexers)
