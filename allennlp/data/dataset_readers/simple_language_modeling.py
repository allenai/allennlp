from typing import Dict, Iterable, List

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register("simple_language_modeling")
class SimpleLanguageModelingDatasetReader(DatasetReader):
    """
    Reads sentences, one per line, for language modeling. This does not handle arbitrarily formatted
    text with sentences spanning multiple lines.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    max_sequence_length: ``int``, optional
        If specified, sentences with more than this number minus two of tokens
        (for the implicit start and end tokens) will be dropped.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None) -> None:
        # Always lazy to handle looping indefinitely.
        super().__init__(True)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length

        print("Creating LMDatasetReader")
        print("max_sequence_length={}".format(max_sequence_length))

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        print('Loading data from {}'.format(file_path))

        with open(file_path) as file:
            all_sentences_raw = file.readlines()

        # Remove sentences longer than the maximum.
        if self._max_sequence_length is not None:
            sentences_raw = [
                sentence for sentence in all_sentences_raw
                if len(self._tokenizer.tokenize(sentence)) <= self._max_sequence_length + 2
            ]
        else:
            sentences_raw = all_sentences_raw

        for sentence in sentences_raw:
            yield self.text_to_instance(sentence)
