import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple

from overrides import overrides
from transformers.tokenization_auto import AutoTokenizer

from allennlp.common.util import sanitize_wordpiece
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
    tokenizer_kwargs: `Dict[str, Any]`
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.

    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        stride: int = 0,
        truncation_strategy: str = "longest_first",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        if "use_fast" not in tokenizer_kwargs:
            tokenizer_kwargs["use_fast"] = True
            # Note: Just because we request a fast tokenizer doesn't mean we get one.

        # Some tokenizers need to know during initialization whether they are going to produce special tokens or
        # not.
        self.tokenizer_with_special_tokens = AutoTokenizer.from_pretrained(
            model_name, add_special_tokens=True, **tokenizer_kwargs
        )
        self.tokenizer_without_special_tokens = AutoTokenizer.from_pretrained(
            model_name, add_special_tokens=False, **tokenizer_kwargs
        )

        self._add_special_tokens = add_special_tokens
        if self._add_special_tokens:
            self.tokenizer = self.tokenizer_with_special_tokens
        else:
            self.tokenizer = self.tokenizer_without_special_tokens

        self._max_length = max_length
        self._stride = stride
        self._truncation_strategy = truncation_strategy

        (
            self.num_added_start_tokens,
            self.num_added_middle_tokens,
            self.num_added_end_tokens,
        ) = self._determine_num_special_tokens_added()

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=self._add_special_tokens,
            max_length=self._max_length,
            stride=self._stride,
            truncation_strategy=self._truncation_strategy,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens.get("offset_mapping"),
        )

        # If we don't have token offsets, try to calculate them ourselves.
        if token_offsets is None:
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

            match_text = text
            token_texts = [
                sanitize_wordpiece(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)
            ]
            token_offsets = [None] * len(token_ids)
            if self.tokenizer.do_lower_case:
                match_text = match_text.lower()
                token_texts = [t.lower() for t in token_texts]

            min_allowed_skipped_whitespace = 3
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            text_index = 0
            token_index = 0
            while text_index < len(match_text) and token_index < len(token_ids):
                token_text = token_texts[token_index]
                token_start_index = match_text.find(token_text, text_index)

                # Did we not find it at all?
                if token_start_index < 0:
                    token_index += 1
                    # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                    allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                    continue

                # Did we jump too far?
                non_whitespace_chars_skipped = sum(
                    1 for c in match_text[text_index:token_start_index] if not c.isspace()
                )
                if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                    # Too many skipped characters. Something is wrong. Ignore this token.
                    token_index += 1
                    # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                    allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                    continue
                allowed_skipped_whitespace = min_allowed_skipped_whitespace

                token_offsets[token_index] = (
                    token_start_index,
                    token_start_index + len(token_text),
                )
                text_index = token_start_index + len(token_text)
                token_index += 1

        tokens = []
        for token_id, token_type_id, offsets in zip(token_ids, token_type_ids, token_offsets):
            if offsets is None or offsets[0] >= offsets[1]:
                token_str = self.tokenizer.convert_ids_to_tokens(
                    token_id, skip_special_tokens=False
                )
                start = None
            else:
                start, end = offsets
                token_str = text[start:end]

            tokens.append(Token(text=token_str, text_id=token_id, type_id=token_type_id, idx=start))
        return tokens

    def intra_word_tokenize(
        self, string_tokens: List[str]
    ) -> Tuple[List[Token], List[Tuple[int, int]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        return self.intra_word_tokenize_sentence_pair(string_tokens, None)[:2]

    def intra_word_tokenize_sentence_pair(
        self, string_tokens_a: List[str], string_tokens_b: Optional[List[str]]
    ) -> Tuple[List[Token], List[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:

        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpieces[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """

        def tokens_from_string_tokens(
            string_tokens: List[str],
        ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
            tokens: List[Token] = []
            offsets: List[Optional[Tuple[int, int]]] = []
            for token_string in string_tokens:
                wordpieces = self.tokenizer_without_special_tokens.encode_plus(
                    token_string,
                    add_special_tokens=False,
                    return_tensors=None,
                    return_offsets_mapping=self.tokenizer_without_special_tokens.is_fast,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                wp_ids, wp_offsets = wordpieces["input_ids"], wordpieces.get("offset_mapping")
                if wp_offsets is None:
                    from allennlp.common.util import sanitize_wordpiece

                    wp_texts = [
                        sanitize_wordpiece(wp)
                        for wp in self.tokenizer_without_special_tokens.convert_ids_to_tokens(
                            wp_ids
                        )
                    ]
                else:
                    wp_texts = [token_string[start:end] for start, end in wp_offsets]

                if len(wp_ids) > 0:
                    offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                    tokens.extend(
                        Token(text=wp_text, text_id=wp_id)
                        for wp_id, wp_text in zip(wp_ids, wp_texts)
                    )
                else:
                    offsets.append(None)
            return tokens, offsets

        dummy_a = 2000
        dummy_b = 3000
        dummy_map: Dict[int, Tuple] = {dummy_a: tokens_from_string_tokens(string_tokens_a)}
        if string_tokens_b is not None:
            dummy_map[dummy_b] = tokens_from_string_tokens(string_tokens_b)

        if string_tokens_b is None:
            dummy_token_ids = self.tokenizer_with_special_tokens.encode_plus(
                [self.tokenizer_with_special_tokens.convert_ids_to_tokens(dummy_a)],
                return_attention_mask=False,
                add_special_tokens=True,
            )
        else:
            dummy_token_ids = self.tokenizer_with_special_tokens.encode_plus(
                self.tokenizer_with_special_tokens.convert_ids_to_tokens(dummy_a),
                self.tokenizer_with_special_tokens.convert_ids_to_tokens(dummy_b),
                return_attention_mask=False,
                add_special_tokens=True,
            )
        dummy_type_ids = dummy_token_ids["token_type_ids"]
        dummy_token_ids = dummy_token_ids["input_ids"]

        tokens_with_special_tokens: List[Token] = []
        offsets_with_special_tokens: List[Optional[Tuple[int, int]]] = []
        for token_id, type_id in zip(dummy_token_ids, dummy_type_ids):
            if token_id in dummy_map:
                tokens, offsets = dummy_map[token_id]
                offsets_with_special_tokens.extend(
                    (
                        offset[0] + len(tokens_with_special_tokens),
                        offset[1] + len(tokens_with_special_tokens),
                    )
                    if offset is not None
                    else None
                    for offset in offsets
                )
                tokens_with_special_tokens.extend(
                    dataclasses.replace(t, type_id=type_id) for t in tokens
                )
                del dummy_map[
                    token_id
                ]  # We can't output the same tokens twice, so we prevent it this way.
            else:
                tokens_with_special_tokens.append(
                    Token(
                        text_id=token_id,
                        text=self.tokenizer.convert_ids_to_tokens(token_id),
                        type_id=type_id,
                    )
                )
        assert len(dummy_map) <= 0

        if string_tokens_b is None:
            return tokens_with_special_tokens, offsets_with_special_tokens, None
        else:
            return (
                tokens_with_special_tokens,
                offsets_with_special_tokens[: len(string_tokens_a)],
                offsets_with_special_tokens[len(string_tokens_a) :],
            )

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

        assert num_start + num_middle + num_end == self.tokenizer.num_special_tokens_to_add(
            pair=True
        )
        assert num_start + num_end == self.tokenizer.num_special_tokens_to_add(pair=False)
        return num_start, num_middle, num_end
