import random
import glob

from overrides import overrides

from allennlp.common.util import START_SYMBOL, END_SYMBOL

from typing import Dict, List, Iterable

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data import Instance, Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN


@DatasetReader.register("lm_dataset_reader")
class LMDatasetReader(DatasetReader):
    def __init__(self,
                 max_sequence_length=None,
                 test=False,
                 queue=None,
                 use_byte_pairs=False,
                 add_suffix_char=False):
        super().__init__(True)
        self._tokenizer = WordTokenizer()
        if use_byte_pairs:
            # TODO(brendanr): Handle this.
            #self._indexers = {
            #    'token': BytePairTokenBatcher(vocab_file, add_suffix_char=add_suffix_char),
            #}
            raise NotImplementedError
        else:
            self._token_indexer = SingleIdTokenIndexer()
            self._character_indexer = ELMoTokenCharactersIndexer()
        self._test = test
        self._queue = queue

        self._max_sequence_length = max_sequence_length

        print("Creating LMDatasetReader")
        print("max_sequence_length={}".format(max_sequence_length))

    def _load_shard(self, next_shard_name):
        print('Loading data from {}'.format(next_shard_name))

        with open(next_shard_name) as f:
            all_sentences_raw = [line for line in f.readlines()]

        # remove sentences longer than the maximum
        if self._max_sequence_length is not None:
            sentences_raw = [
                sentence for sentence in all_sentences_raw
                if len(self._tokenizer.tokenize(sentence)) <= self._max_sequence_length + 2
            ]
        else:
            sentences_raw = all_sentences_raw

        for sentence in sentences_raw:
            yield self.text_to_instance(sentence)


    def _get_next_shard_name(self):
        """Randomly select a file."""
        if self._queue is not None:
            # just get the next shard from the queue
            return self._queue.get()

        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                shard_name = None
            else:
                shard_name = self._all_shards.pop()
        else:
            # just pick a random shard
            if len(self._shards_to_choose) == 0:
                self._shards_to_choose = list(self._all_shards)
                random.shuffle(self._shards_to_choose)
            shard_name = self._shards_to_choose.pop()

        return shard_name

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        raw_tokenized = self._tokenizer.tokenize(sentence)
        tokenized = [Token(START_SYMBOL)] + raw_tokenized + [Token(END_SYMBOL)]
        forward_targets = raw_tokenized + [Token(END_SYMBOL)]
        # TODO(brendanr): Using padding token here may be breaking API boundaries. Are there alternatives?
        backward_targets = [Token(DEFAULT_PADDING_TOKEN), Token(START_SYMBOL)] + raw_tokenized

        return_instance = Instance({
            'forward_targets': TextField(forward_targets, {"tokens": self._token_indexer}),
            'backward_targets': TextField(backward_targets, {"tokens": self._token_indexer}),
            # TODO(brendanr): Place these together under a single text field as is standard in AllenNLP?
            'tokens': TextField(tokenized, {"tokens": self._token_indexer}),
            'characters': TextField(tokenized, {"characters": self._character_indexer})
        })
        return return_instance

    @overrides
    def _read(self, file_prefix):
        self._all_shards = glob.glob(file_prefix)
        self._shards_to_choose = []

        while True:
            next_shard_name = self._get_next_shard_name()
            if next_shard_name is None:
                break
            instances = self._load_shard(next_shard_name)

            for instance in instances:
                yield instance
