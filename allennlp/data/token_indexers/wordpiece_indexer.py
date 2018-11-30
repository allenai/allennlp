# pylint: disable=no-self-use
from typing import Dict, List, Callable
import logging

from overrides import overrides

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)

# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.

class WordpieceIndexer(TokenIndexer[int]):
    """
    A token indexer that does the wordpiece-tokenization (e.g. for BERT embeddings).
    If you are using one of the pretrained BERT models, you'll want to use the ``PretrainedBertIndexer``
    subclass rather than this base class.

    Parameters
    ----------
    vocab : ``Dict[str, int]``
        The mapping {wordpiece -> id}.  Note this is not an AllenNLP ``Vocabulary``.
    wordpiece_tokenizer : ``Callable[[str], List[str]]``
        A function that does the actual tokenization.
    namespace : str, optional (default: "wordpiece")
        The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
        will be loaded.
    use_starting_offsets : bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Currently any inputs longer than this
        will be truncated. If this behavior is undesirable to you, you should
        consider filtering them out in your dataset reader.
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 wordpiece_tokenizer: Callable[[str], List[str]],
                 namespace: str = "wordpiece",
                 use_starting_offsets: bool = False,
                 max_pieces: int = 512) -> None:
        self.vocab = vocab

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertSimpleWordSplitter``,
        # and this token indexer handles the second.
        self.wordpiece_tokenizer = wordpiece_tokenizer

        self._namespace = namespace
        self._added_to_vocabulary = False
        self.max_pieces = max_pieces
        self.use_starting_offsets = use_starting_offsets

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.vocab.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        text_tokens: List[int] = []
        offsets = []
        # For initial offsets, start at 0; otherwise, start at -1
        offset = 0 if self.use_starting_offsets else -1

        for token in tokens:
            wordpieces = self.wordpiece_tokenizer(token.text)
            wordpiece_ids = [self.vocab[token] for token in wordpieces]

            # truncate and pray
            if len(text_tokens) + len(wordpiece_ids) > self.max_pieces:
                # TODO(joelgrus): figure out a better way to handle this
                logger.warning(f"Too many wordpieces, truncating: {[token.text for token in tokens]}")
                break

            # For initial offsets, the current value of ``offset`` is the start of
            # the current wordpiece, so add it to ``offsets`` and then increment it.
            if self.use_starting_offsets:
                offsets.append(offset)
                offset += len(wordpiece_ids)
            # For final offsets, the current value of ``offset`` is the end of
            # the previous wordpiece, so increment it and then add it to ``offsets``.
            else:
                offset += len(wordpiece_ids)
                offsets.append(offset)
            text_tokens.extend(wordpiece_ids)

        # add mask according to the original tokens,
        # because calling util.get_text_field_mask on the
        # "byte pair" tokens will produce the wrong shape
        mask = [1 for _ in offsets]

        return {
                index_name: text_tokens,
                f"{index_name}-offsets": offsets,
                "mask": mask
        }

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-offsets", "mask"]


@TokenIndexer.register("bert-pretrained")
class PretrainedBertIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``, optional (default = None)
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Currently any inputs longer than this
        will be truncated. If this behavior is undesirable to you, you should
        consider filtering them out in your dataset reader.
    """
    def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 max_pieces: int = 512) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lowercase)
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         namespace="bert",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces)
