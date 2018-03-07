from typing import Dict
import logging

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("language_modeling")
class LanguageModelingReader(DatasetReader):
    """
    Reads a text file and converts it into a ``Dataset`` suitable for training a language model.

    Note that there's one issue that needs to be fixed before this is actually usable for language
    modeling - the way start and end tokens for sentences are handled is not correct; we need to
    add a sentence splitter before this will be done right.

    Parameters
    ----------
    tokens_per_instance : ``int``, optional (default=``None``)
        If this is ``None``, we will have each training instance be a single sentence.  If this is
        not ``None``, we will instead take all sentences, including their start and stop tokens,
        line them up, and split the tokens into groups of this number, for more efficient training
        of the language model.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    """
    def __init__(self,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokens_per_instance = tokens_per_instance

        # No matter how you want to represent the input, we'll always represent the output as a
        # single token id.  This code lets you learn a language model that concatenates word
        # embeddings with character-level encoders, in order to predict the word token that comes
        # next.
        self._output_indexer: Dict[str, TokenIndexer] = None
        for name, indexer in self._token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                self._output_indexer = {name: indexer}
                break
        else:
            self._output_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as text_file:
            instance_strings = text_file.readlines()

        if self._tokens_per_instance is not None:
            all_text = " ".join([x.replace("\n", " ").strip() for x in instance_strings])
            tokenized_text = self._tokenizer.tokenize(all_text)
            num_tokens = self._tokens_per_instance + 1
            tokenized_strings = []
            logger.info("Creating dataset from all text in file: %s", file_path)
            for index in Tqdm.tqdm(range(0, len(tokenized_text) - num_tokens, num_tokens - 1)):
                tokenized_strings.append(tokenized_text[index:(index + num_tokens)])
        else:
            tokenized_strings = [self._tokenizer.tokenize(s) for s in instance_strings]

        for tokenized_string in tokenized_strings:
            input_field = TextField(tokenized_string[:-1], self._token_indexers)
            output_field = TextField(tokenized_string[1:], self._output_indexer)
            yield Instance({'input_tokens': input_field,
                            'output_tokens': output_field})

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokenized_string[:-1], self._token_indexers)
        output_field = TextField(tokenized_string[1:], self._output_indexer)
        return Instance({'input_tokens': input_field, 'output_tokens': output_field})

    @classmethod
    def from_params(cls, params: Params) -> 'LanguageModelingReader':
        tokens_per_instance = params.pop_int('tokens_per_instance', None)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return LanguageModelingReader(tokens_per_instance=tokens_per_instance,
                                      tokenizer=tokenizer,
                                      token_indexers=token_indexers,
                                      lazy=lazy)
