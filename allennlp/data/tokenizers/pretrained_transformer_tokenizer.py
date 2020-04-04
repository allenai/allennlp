import logging
from typing import Any, Dict, List, Optional, Tuple

from allennlp.common.util import sanitize_wordpiece
from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@Tokenizer.register("pretrained_transformer")
class PretrainedTransformerTokenizer(Tokenizer):
    """
    A `PretrainedTransformerTokenizer` uses a model from HuggingFace's
    `transformers` library to tokenize some input text.  This often means wordpieces
    (where `'AllenNLP is awesome'` might get split into `['Allen', '##NL', '##P', 'is',
    'awesome']`), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.

    We take a model name as an input parameter, which we will pass to
    `AutoTokenizer.from_pretrained`.

    We also add special tokens relative to the pretrained model and truncate the sequences.

    This tokenizer also indexes tokens and adds the indexes to the `Token` fields so that
    they can be picked up by `PretrainedTransformerIndexer`.

    Registered as a `Tokenizer` with name "pretrained_transformer".

    # Parameters

    model_name : `str`
        The name of the pretrained wordpiece tokenizer to use.
    add_special_tokens : `bool`, optional, (default=True)
        If set to `True`, the sequences will be encoded with the special tokens relative
        to their model.
    max_length : `int`, optional (default=None)
        If set to a number, will limit the total sequence returned so that it has a maximum length.
        If there are overflowing tokens, those will be added to the returned dictionary
    stride : `int`, optional (default=0)
        If set to a number along with max_length, the overflowing tokens returned will contain some tokens
        from the main sequence returned. The value of this argument defines the number of additional tokens.
    truncation_strategy : `str`, optional (default='longest_first')
        String selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
        starting from the longest one at each token (when there is a pair of input sequences)
        - 'only_first': Only truncate the first sequence
        - 'only_second': Only truncate the second sequence
        - 'do_not_truncate': Do not truncate (raise an error if the input sequence is longer than max_length)
    calculate_character_offsets : `bool`, optional (default=False)
        Attempts to reconstruct character offsets for the instances of Token that this tokenizer produces.
    tokenizer_kwargs: 'Dict[str, Any]'
        Dictionary with additional arguments for `AutoTokenizer.from_pretrained`.

    Argument descriptions are from
    https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691
    """

    def __init__(
        self,
        model_name: str,
        add_special_tokens: bool = True,
        max_length: int = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        calculate_character_offsets: bool = False,
        tokenizer_kwargs: Dict[str, Any] = None,
    ) -> None:
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        # Huggingface tokenizers have different ways of remembering whether they lowercase or not. Detecting it
        # this way seems like the least brittle way to do it.
        tokenized = self.tokenizer.tokenize(
            "A"
        )  # Use a single character that won't be cut into word pieces.
        detokenized = " ".join(tokenized)
        self._tokenizer_lowercases = "a" in detokenized

        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncation_strategy
        self._calculate_character_offsets = calculate_character_offsets

        (
            self.num_added_start_tokens,
            self.num_added_middle_tokens,
            self.num_added_end_tokens,
        ) = self._determine_num_special_tokens_added()

    def _tokenize(self, sentence_1: str, sentence_2: str = None):
        """
        This method works on both sentence and sentence pair.
        """

        encoded_tokens = self.tokenizer.encode_plus(
            text=sentence_1,
            text_pair=sentence_2,
            add_special_tokens=self._add_special_tokens,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=None,
            return_token_type_ids=True,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids = encoded_tokens["input_ids"], encoded_tokens["token_type_ids"]

        tokens = []
        for token_id, token_type_id in zip(token_ids, token_type_ids):
            token_str = self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False)
            tokens.append(Token(text=token_str, text_id=token_id, type_id=token_type_id))

        if self._calculate_character_offsets:
            # The huggingface tokenizers produce tokens that may or may not be slices from the
            # original text.  Differences arise from lowercasing, Unicode normalization, and other
            # kinds of normalization, as well as special characters that are included to denote
            # various situations, such as "##" in BERT for word pieces from the middle of a word, or
            # "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.

            # This code attempts to calculate character offsets while being tolerant to these
            # differences. It scans through the text and the tokens in parallel, trying to match up
            # positions in both. If it gets out of sync, it backs off to not adding any token
            # indices, and attempts to catch back up afterwards. This procedure is approximate.
            # Don't rely on precise results, especially in non-English languages that are far more
            # affected by Unicode normalization.

            whole_text = sentence_1
            if sentence_2 is not None:
                whole_text += sentence_2  # Calculating character offsets with sentence pairs is sketchy at best.
            if self._tokenizer_lowercases:
                whole_text = whole_text.lower()

            min_allowed_skipped_whitespace = 3
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            text_index = 0
            token_index = 0
            while text_index < len(whole_text) and token_index < len(tokens):
                token_text = tokens[token_index].text
                if self._tokenizer_lowercases:
                    token_text = token_text.lower()
                token_text = sanitize_wordpiece(token_text)
                token_start_index = whole_text.find(token_text, text_index)

                # Did we not find it at all?
                if token_start_index < 0:
                    token_index += 1
                    # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                    allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                    continue

                # Did we jump too far?
                non_whitespace_chars_skipped = sum(
                    1 for c in whole_text[text_index:token_start_index] if not c.isspace()
                )
                if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                    # Too many skipped characters. Something is wrong. Ignore this token.
                    token_index += 1
                    # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                    allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                    continue
                allowed_skipped_whitespace = min_allowed_skipped_whitespace

                tokens[token_index].idx = token_start_index
                text_index = token_start_index + len(token_text)
                token_index += 1

        return tokens

    def tokenize_sentence_pair(self, sentence_1: str, sentence_2: str) -> List[Token]:
        """
        This methods properly handles a pair of sentences.
        """
        return self._tokenize(sentence_1, sentence_2)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        Refer to the `tokenize_sentence_pair` method if you have a sentence pair.
        """
        return self._tokenize(text)

    def intra_word_tokenize(self, tokens: List[str]) -> Tuple[List[Token], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpices[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        wordpieces, offsets, cumulative = self.intra_word_tokenize_in_id(
            tokens, self.num_added_start_tokens
        )
        wp_tokens = self.ids_to_tokens(wordpieces)
        assert cumulative + self.num_added_end_tokens == len(wp_tokens)
        return wp_tokens, offsets

    def intra_word_tokenize_sentence_pair(
        self, tokens_a: List[str], tokens_b: List[str]
    ) -> Tuple[List[Token], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpices[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        wordpieces_a, offsets_a, cumulative = self.intra_word_tokenize_in_id(
            tokens_a, self.num_added_start_tokens
        )
        wordpieces_b, offsets_b, cumulative = self.intra_word_tokenize_in_id(
            tokens_b, cumulative + self.num_added_middle_tokens
        )
        wp_tokens = self.ids_to_tokens(wordpieces_a, wordpieces_b)
        assert cumulative + self.num_added_end_tokens == len(wp_tokens)
        return wp_tokens, offsets_a, offsets_b

    def intra_word_tokenize_in_id(
        self, tokens: List[str], starting_offset: int = 0
    ) -> Tuple[List[int], List[Tuple[int, int]], int]:
        """
        Similar to `intra_word_tokenize()`, except:
        (a) returns wordpiece IDs in the vocab instead of `Token`s;
        (b) only takes a single sequence; and
        (c) does not insert special tokens.
        """
        wordpieces: List[int] = []
        offsets = []
        cumulative = starting_offset
        for token in tokens:
            subword_wordpieces = self.tokenizer.encode(token, add_special_tokens=False)
            if len(subword_wordpieces) == 0:
                subword_wordpieces = [self.tokenizer.unk_token_id]

            wordpieces.extend(subword_wordpieces)

            start_offset = cumulative
            cumulative += len(subword_wordpieces)
            end_offset = cumulative - 1  # inclusive
            offsets.append((start_offset, end_offset))
        return wordpieces, offsets, cumulative

    def ids_to_tokens(
        self, token_ids_a: List[int], token_ids_b: Optional[List[int]] = None
    ) -> List[Token]:
        """
        Convert one or two sequences of token IDs to `Token`s while adding special tokens.
        """
        text_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_a, token_ids_b)
        type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_a, token_ids_b)

        tokens = [
            Token(self.tokenizer.convert_ids_to_tokens(text_id), text_id=text_id, type_id=type_id)
            for text_id, type_id in zip(text_ids, type_ids)
        ]
        return tokens

    def _determine_num_special_tokens_added(self) -> Tuple[int, int, int]:
        """
        Determines the number of tokens `tokenizer` adds to a sequence or sequence pair
        in the start, middle, and end.

        # Returns

        The number of tokens (`int`) that are inserted in the start, middle, and end of a sequence.
        """
        # Uses a slightly higher index to avoid tokenizer doing special things to lower-indexed
        # tokens which might be special.
        dummy_a = [1000]
        dummy_b = [2000]
        inserted = self.tokenizer.build_inputs_with_special_tokens(dummy_a, dummy_b)

        num_start = num_middle = num_end = 0
        seen_dummy_a = False
        seen_dummy_b = False
        for idx in inserted:
            if idx == dummy_a[0]:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                continue

            if idx == dummy_b[0]:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_b = True
                continue

            if not seen_dummy_a:
                num_start += 1
            elif not seen_dummy_b:
                num_middle += 1
            else:
                num_end += 1

        assert num_start + num_middle + num_end == self.tokenizer.num_added_tokens(pair=True)
        assert num_start + num_end == self.tokenizer.num_added_tokens(pair=False)
        return num_start, num_middle, num_end
