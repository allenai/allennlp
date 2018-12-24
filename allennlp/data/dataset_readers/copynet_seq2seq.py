import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, CopyMapField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("copynet_seq2seq")
class CopyNetDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string><tab><target_sequence_string>.

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the CopyMapField.
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    """

    def __init__(self,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

    @staticmethod
    def _read_line(line_num: int, line: str) -> Tuple[Optional[str], Optional[str]]:
        line = line.strip("\n")
        if not line:
            return None, None
        line_parts = line.split('\t')
        if len(line_parts) != 2:
            raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
        source_sequence, target_sequence = line_parts
        return source_sequence, target_sequence

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                source_sequence, target_sequence = self._read_line(line_num, line)
                if not source_sequence:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence)

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an `Instance`.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional

        Returns
        -------
        Instance
            An Instance containing at least the following fields:

            - `source_tokens`: a `TextField` containing the tokenized source sentence,
               including the `START_SYMBOL` and `END_SYMBOL`.
               This will result in a tensor of shape `(batch_size, source_length)`.

            - `source_token_ids`: an `ArrayField` of size `(batch_size, trimmed_source_length)`
              that contains an ID for each token in the source sentence. Tokens that
              match at the lowercase level will share the same ID. If `target_tokens`
              is passed as well, these IDs will also correspond to the `target_token_ids`
              field, i.e. any tokens that match at the lowercase level in both
              the source and target sentences will share the same ID. Note that these IDs
              have no correlation with the token indices from the corresponding
              vocabulary namespaces.

            - `source_to_target`: a `CopyMapField` that keeps track of the index
              of the target token that matches each token in the source sentence.
              When there is no matching target token, the OOV index is used.
              This will result in a tensor of shape `(batch_size, trimmed_source_length)`.

            - `metadata`: a `MetadataField` which contains the source tokens and
              potentially target tokens as lists of strings.

            When `target_string` is passed, the instance will also contain these fields:

            - `target_tokens`: a `TextField` containing the tokenized target sentence,
              including the `START_SYMBOL` and `END_SYMBOL`. This will result in
              a tensor of shape `(batch_size, target_length)`.

            - `target_token_ids`: an `ArrayField` of size `(batch_size, target_length)`.
              This is calculated in the same way as `source_token_ids`.

        Notes
        -----
        By `source_length` we are referring to the number of tokens in the source
        sentence including the `START_SYMBOL` and `END_SYMBOL`, while
        `trimmed_source_length` refers to the number of tokens in the source sentence
        *excluding* the `START_SYMBOL` and `END_SYMBOL`, i.e.
        `trimmed_source_length = source_length - 2`.

        On the other hand, `target_length` is the number of tokens in the target sentence
        *including* the `START_SYMBOL` and `END_SYMBOL`.

        In the context where there is a `batch_size` dimension, the above refer
        to the maximum of their individual values across the batch.
        """
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = CopyMapField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
