import logging
import os
from typing import Iterable

from pytorch_pretrained_bert.tokenization import load_vocab, PRETRAINED_VOCAB_ARCHIVE_MAP, VOCAB_NAME
from pytorch_pretrained_bert.file_utils import cached_path

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN


@Vocabulary.register("wordpiece")
class WordpieceVocabulary(Vocabulary):
    def __init__(self,
                 pretrained_model: str,   # Can point to pretrained model or vocab dir/file
                 namespace: str = 'wordpiece',
                 **kwargs):

        super().__init__(**kwargs)
        pretrained_model_name = pretrained_model
        # Copied from pytorch_pretrained_bert.tokenization BertTokenizer code
        if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
        else:
            vocab_file = pretrained_model_name
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file)
        except FileNotFoundError:
            logging.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logging.info("loading vocabulary file {}".format(vocab_file))
        else:
            logging.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        wordpiece_vocab = load_vocab(resolved_vocab_file)
        for word, idx in wordpiece_vocab.items():
            if idx == 0:
                word = DEFAULT_PADDING_TOKEN
            self._token_to_index[namespace][word] = idx
            self._index_to_token[namespace][idx] = word

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        pretrained_model = params.pop('pretrained_model')
        namespace = params.pop('namespace', 'wordpiece')
        return WordpieceVocabulary(pretrained_model=pretrained_model,
                                   namespace=namespace)
    @classmethod
    def from_files(cls, directory: str) -> Vocabulary:
        return Vocabulary.from_files(directory)