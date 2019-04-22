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

# This is the default list of tokens that should not be lowercased.
_NEVER_LOWERCASE = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']


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
    do_lowercase : ``bool``, optional (default=``False``)
        Should we lowercase the provided tokens before getting the indices?
        You would need to do this if you are using an -uncased BERT model
        but your DatasetReader is not lowercasing tokens (which might be the
        case if you're also using other embeddings based on cased tokens).
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    separator_token : ``str``, optional (default=``[SEP]``)
        This token indicates the segments in the sequence.
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 wordpiece_tokenizer: Callable[[str], List[str]],
                 namespace: str = "wordpiece",
                 use_starting_offsets: bool = False,
                 max_pieces: int = 512,
                 do_lowercase: bool = False,
                 never_lowercase: List[str] = None,
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 separator_token: str = "[SEP]") -> None:
        self.vocab = vocab

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertBasicWordSplitter``,
        # and this token indexer handles the second.
        self.wordpiece_tokenizer = wordpiece_tokenizer

        self._namespace = namespace
        self._added_to_vocabulary = False
        self.max_pieces = max_pieces
        self.use_starting_offsets = use_starting_offsets
        self._do_lowercase = do_lowercase

        if never_lowercase is None:
            # Use the defaults
            self._never_lowercase = set(_NEVER_LOWERCASE)
        else:
            self._never_lowercase = set(never_lowercase)

        # Convert the start_tokens and end_tokens to wordpiece_ids
        self._start_piece_ids = [vocab[wordpiece]
                                 for token in (start_tokens or [])
                                 for wordpiece in wordpiece_tokenizer(token)]
        self._end_piece_ids = [vocab[wordpiece]
                               for token in (end_tokens or [])
                               for wordpiece in wordpiece_tokenizer(token)]

        # Convert the separator_token to wordpiece_ids
        self._separator_ids = [vocab[wordpiece]
                               for wordpiece in wordpiece_tokenizer(separator_token)]

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

        # The array of wordpiece_ids to return.
        # Start with a copy of the start_piece_ids
        wordpiece_ids: List[int] = self._start_piece_ids[:]

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # If we're using initial offsets, we want to start at offset = len(text_tokens)
        # so that the first offset is the index of the first wordpiece of tokens[0].
        # Otherwise, we want to start at len(text_tokens) - 1, so that the "previous"
        # offset is the last wordpiece of "tokens[-1]".
        offset = len(wordpiece_ids) if self.use_starting_offsets else len(wordpiece_ids) - 1

        for token in tokens:
            # Lowercase if necessary
            text = (token.text.lower()
                    if self._do_lowercase and token.text not in self._never_lowercase
                    else token.text)
            token_wordpiece_ids = [self.vocab[wordpiece]
                                   for wordpiece in self.wordpiece_tokenizer(text)]
            # If we have enough room to add these ids *and also* the end_token ids.
            if len(wordpiece_ids) + len(token_wordpiece_ids) + len(self._end_piece_ids) <= self.max_pieces:
                # For initial offsets, the current value of ``offset`` is the start of
                # the current wordpiece, so add it to ``offsets`` and then increment it.
                if self.use_starting_offsets:
                    offsets.append(offset)
                    offset += len(token_wordpiece_ids)
                # For final offsets, the current value of ``offset`` is the end of
                # the previous wordpiece, so increment it and then add it to ``offsets``.
                else:
                    offset += len(token_wordpiece_ids)
                    offsets.append(offset)
                # And add the token_wordpiece_ids to the output list.
                wordpiece_ids.extend(token_wordpiece_ids)
            else:
                # TODO(joelgrus): figure out a better way to handle this
                logger.warning(f"Too many wordpieces, truncating: {[token.text for token in tokens]}")
                break

        # By construction, we still have enough room to add the end_token ids.
        wordpiece_ids.extend(self._end_piece_ids)
        # Constructing `token_type_ids` by `self._separator`
        token_type_ids = _get_token_type_ids(wordpiece_ids,
                                             self._separator_ids)

        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {
                index_name: wordpiece_ids,
                f"{index_name}-offsets": offsets,
                f"{index_name}-type-ids": token_type_ids,
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
        return [index_name, f"{index_name}-offsets", f"{index_name}-type-ids", "mask"]


@TokenIndexer.register("bert-pretrained")
class PretrainedBertIndexer(WordpieceIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
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
    never_lowercase: ``List[str]``, optional
        Tokens that should never be lowercased. Default is
        ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'].
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
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512) -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")

        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)
        super().__init__(vocab=bert_tokenizer.vocab,
                         wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                         namespace="bert",
                         use_starting_offsets=use_starting_offsets,
                         max_pieces=max_pieces,
                         do_lowercase=do_lowercase,
                         never_lowercase=never_lowercase,
                         start_tokens=["[CLS]"],
                         end_tokens=["[SEP]"],
                         separator_token="[SEP]")


def _get_token_type_ids(wordpiece_ids: List[int],
                        separator_ids: List[int]) -> List[int]:
    num_wordpieces = len(wordpiece_ids)
    token_type_ids: List[int] = []
    type_id = 0
    cursor = 0
    while cursor < num_wordpieces:
        # check length
        if num_wordpieces - cursor < len(separator_ids):
            token_type_ids.extend(type_id
                                  for _ in range(num_wordpieces - cursor))
            cursor += num_wordpieces - cursor
        # check content
        # when it is a separator
        elif all(wordpiece_ids[cursor + index] == separator_id
                 for index, separator_id in enumerate(separator_ids)):
            token_type_ids.extend(type_id for _ in separator_ids)
            type_id += 1
            cursor += len(separator_ids)
        # when it is not
        else:
            cursor += 1
            token_type_ids.append(type_id)
    return token_type_ids
