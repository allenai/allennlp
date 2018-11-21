from typing import Dict, List
import logging

from overrides import overrides

from pytorch_pretrained_bert.tokenization import WordpieceTokenizer, BertTokenizer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)


class BertIndexer(TokenIndexer[int]):
    """
    A token indexer that does the wordpiece-tokenization required for BERT embeddings.
    Most likely you are using one of the pretrained BERT models, in which case
    you'll want to use the ``PretrainedBertIndexer`` subclass rather than this base class.

    Parameters
    ----------
    vocab: ``Dict[str, int]``
        The mapping {wordpiece -> id}.  Note this is not an AllenNLP ``Vocabulary``.
    wordpiece_tokenizer: ``WordpieceTokenizer``
        The class that does the actual tokenization.
    namespace: str, optional (default: "bert")
        The namespace in the AllenNLP ``Vocabulary`` into which the wordpieces
        will be loaded.
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Currently any inputs longer than this
        will be truncated. If this behavior is undesirable to you, you should
        consider filtering them out in your dataset reader.
    """
    def __init__(self,
                 vocab: Dict[str, int],
                 wordpiece_tokenizer: WordpieceTokenizer,
                 namespace: str = "bert",
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

        text_tokens = []
        offsets = []
        offset = -1

        for token in tokens:
            wordpieces = self.wordpiece_tokenizer.tokenize(token.text)
            wordpiece_ids = [self.vocab[token] for token in wordpieces]

            # truncate and pray
            if len(text_tokens) + len(wordpiece_ids) > self.max_pieces:
                # TODO(joelgrus): figure out a better way to handle this
                logger.warning(f"Too many wordpieces, truncating: {[token.text for token in tokens]}")
                break

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
                "mask": mask,

                # This is a really bad hack to avoid triggering the
                # "all indices have the same length" logic in TextField.get_padding_lengths
                # TODO(joelgrus): fix the logic in TextField.get_padding_lengths and remove this
                #"_ignoreme": [0, 0] if len(mask) == 1 else [0]
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


@TokenIndexer.register("bert-pretrained")
class PretrainedBertIndexer(BertIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT models.

    Parameters
    ----------
    pretrained_model: ``str``, optional (default = None)
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    """
    def __init__(self,
                 pretrained_model: str,
                 do_lowercase: bool = True) -> None:
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lowercase)
        super().__init__(bert_tokenizer.vocab, bert_tokenizer.wordpiece_tokenizer)
