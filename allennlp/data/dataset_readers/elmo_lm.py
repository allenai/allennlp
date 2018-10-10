import random
import glob
from typing import Dict, Iterable

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register("elmo_lm_dataset_reader")
class ElmoLMDatasetReader(DatasetReader):
    """
    Reads sentences, one per line, from sharded files for language modeling.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sentences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer(), "characters": ELMoTokenCharactersIndexer()}``.
    max_sequence_length: ``int``, optional
        If specified sentences with more than this number of tokens will be dropped.
    loop_indefinitely: ``bool``, optional
        Whether to re-read the shards indefinitely. Defaults to true.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = None,
                 loop_indefinitely: bool = True) -> None:
        # Always lazy to handle looping indefinitely.
        super().__init__(True)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
                "tokens": SingleIdTokenIndexer(),
                "characters": ELMoTokenCharactersIndexer()
        }
        self._max_sequence_length = max_sequence_length
        self._loop_indefinitely = loop_indefinitely
        self._all_shards = None
        self._shards_to_choose = []

        print("Creating LMDatasetReader")
        print("max_sequence_length={}".format(max_sequence_length))

    def _load_shard(self, next_shard_name: str) -> Iterable[Instance]:
        """Iterate over the instances in the given shard."""

        print('Loading data from {}'.format(next_shard_name))

        with open(next_shard_name) as shard:
            all_sentences_raw = [line for line in shard.readlines()]

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


    def _get_next_shard_name(self) -> str:
        """Randomly select a file."""

        if not self._loop_indefinitely:
            if not self._all_shards:
                # We've loaded all the data. This will propagate up to _read and stop iterating.
                shard_name = None
            else:
                shard_name = self._all_shards.pop()
        else:
            # Just pick a random shard.
            if not self._shards_to_choose:
                self._shards_to_choose = list(self._all_shards)
                random.shuffle(self._shards_to_choose)
            shard_name = self._shards_to_choose.pop()

        return shard_name

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        # pylint: disable=arguments-differ
        tokenized = self._tokenizer.tokenize(sentence)
        return_instance = Instance({
                'source': TextField(tokenized, self._token_indexers),
        })
        return return_instance

    @overrides
    def _read(self, file_prefix: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        self._all_shards = glob.glob(file_prefix)

        while True:
            next_shard_name = self._get_next_shard_name()
            if next_shard_name is None:
                break
            instances = self._load_shard(next_shard_name)

            for instance in instances:
                yield instance
