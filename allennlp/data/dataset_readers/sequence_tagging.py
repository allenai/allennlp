from typing import Dict, List
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)

DEFAULT_WORD_TAG_DELIMITER = "###"


@DatasetReader.register("sequence_tagging")
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    ```
    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n
    ```

    and converts it into a `Dataset` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Registered as a `DatasetReader` with name "sequence_tagging".

    # Parameters

    word_tag_delimiter: `str`, optional (default=`"###"`)
        The text that separates each WORD from its TAG.
    token_delimiter: `str`, optional (default=`None`)
        The text that separates each WORD-TAG pair from the next pair. If `None`
        then the line will just be split on whitespace.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """

    def __init__(
        self,
        word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
        token_delimiter: str = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_tag_delimiter = word_tag_delimiter
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:

            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")

                # skip blank lines
                if not line:
                    continue

                tokens_and_tags = [
                    pair.rsplit(self._word_tag_delimiter, 1)
                    for pair in line.split(self._token_delimiter)
                ]
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]
                yield self.text_to_instance(tokens, tags)

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        return Instance(fields)
