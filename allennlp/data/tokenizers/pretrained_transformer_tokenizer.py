import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable

from overrides import overrides
from transformers import PreTrainedTokenizer

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
    add_special_tokens : `bool`, optional, (default=`True`)
        If set to `True`, the sequences will be encoded with the special tokens relative
        to their model.
    max_length : `int`, optional (default=`None`)
        If set to a number, will limit the total sequence returned so that it has a maximum length.
        If there are overflowing tokens, those will be added to the returned dictionary
    stride : `int`, optional (default=`0`)
        If set to a number along with max_length, the overflowing tokens returned will contain some tokens
        from the main sequence returned. The value of this argument defines the number of additional tokens.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
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
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        tokenizer_kwargs.setdefault("use_fast", True)
        # Note: Just because we request a fast tokenizer doesn't mean we get one.

        from allennlp.common import cached_transformers

        self.tokenizer = cached_transformers.get_tokenizer(
            model_name, add_special_tokens=False, **tokenizer_kwargs
        )

        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._stride = stride

        self._tokenizer_lowercases = self.tokenizer_lowercases(self.tokenizer)

        try:
            self._reverse_engineer_special_tokens("a", "b", model_name, tokenizer_kwargs)
        except AssertionError:
            # For most transformer models, "a" and "b" work just fine as dummy tokens.  For a few,
            # they don't, and so we use "1" and "2" instead.
            self._reverse_engineer_special_tokens("1", "2", model_name, tokenizer_kwargs)

    def _reverse_engineer_special_tokens(
        self,
        token_a: str,
        token_b: str,
        model_name: str,
        tokenizer_kwargs: Optional[Dict[str, Any]],
    ):
        # storing the special tokens
        self.sequence_pair_start_tokens = []
        self.sequence_pair_mid_tokens = []
        self.sequence_pair_end_tokens = []
        # storing token type ids for the sequences
        self.sequence_pair_first_token_type_id = None
        self.sequence_pair_second_token_type_id = None

        # storing the special tokens
        self.single_sequence_start_tokens = []
        self.single_sequence_end_tokens = []
        # storing token type id for the sequence
        self.single_sequence_token_type_id = None

        # Reverse-engineer the tokenizer for two sequences
        from allennlp.common import cached_transformers

        tokenizer_with_special_tokens = cached_transformers.get_tokenizer(
            model_name, add_special_tokens=True, **(tokenizer_kwargs or {})
        )
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            token_b,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        dummy_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        assert dummy_a in dummy_output["input_ids"]
        dummy_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]
        assert dummy_b in dummy_output["input_ids"]
        assert dummy_a != dummy_b

        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.sequence_pair_first_token_type_id is None
                    or self.sequence_pair_first_token_type_id == token_type_id
                ), "multiple different token type ids found for the first sequence"
                self.sequence_pair_first_token_type_id = token_type_id
                continue

            if token_id == dummy_b:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_b = True
                assert (
                    self.sequence_pair_second_token_type_id is None
                    or self.sequence_pair_second_token_type_id == token_type_id
                ), "multiple different token type ids found for the second sequence"
                self.sequence_pair_second_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)

        assert (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=True)

        # Reverse-engineer the tokenizer for one sequence
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        seen_dummy_a = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.single_sequence_token_type_id is None
                    or self.single_sequence_token_type_id == token_type_id
                ), "multiple different token type ids found for the sequence"
                self.single_sequence_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)

        assert (
            len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=False)

    @staticmethod
    def tokenizer_lowercases(tokenizer: PreTrainedTokenizer) -> bool:
        # Huggingface tokenizers have different ways of remembering whether they lowercase or not. Detecting it
        # this way seems like the least brittle way to do it.
        tokenized = tokenizer.tokenize(
            "A"
        )  # Use a single character that won't be cut into word pieces.
        detokenized = " ".join(tokenized)
        return "a" in detokenized

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        max_length = self._max_length
        if max_length is not None and not self._add_special_tokens:
            max_length += self.num_special_tokens_for_sequence()

        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=max_length,
            stride=self._stride,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids, special_tokens_mask, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens["special_tokens_mask"],
            encoded_tokens.get("offset_mapping"),
        )

        # If we don't have token offsets, try to calculate them ourselves.
        if token_offsets is None:
            token_offsets = self._estimate_character_indices(text, token_ids)

        tokens = []
        for token_id, token_type_id, special_token_mask, offsets in zip(
            token_ids, token_type_ids, special_tokens_mask, token_offsets
        ):
            # In `special_tokens_mask`, 1s indicate special tokens and 0s indicate regular tokens.
            # NOTE: in transformers v3.4.0 (and probably older versions) the docstring
            # for `encode_plus` was incorrect as it had the 0s and 1s reversed.
            # https://github.com/huggingface/transformers/pull/7949 fixed this.
            if not self._add_special_tokens and special_token_mask == 1:
                continue

            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets

            tokens.append(
                Token(
                    text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )

        return tokens

    def _estimate_character_indices(
        self, text: str, token_ids: List[int]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        The huggingface tokenizers produce tokens that may or may not be slices from the
        original text.  Differences arise from lowercasing, Unicode normalization, and other
        kinds of normalization, as well as special characters that are included to denote
        various situations, such as "##" in BERT for word pieces from the middle of a word, or
        "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.

        This code attempts to calculate character offsets while being tolerant to these
        differences. It scans through the text and the tokens in parallel, trying to match up
        positions in both. If it gets out of sync, it backs off to not adding any token
        indices, and attempts to catch back up afterwards. This procedure is approximate.
        Don't rely on precise results, especially in non-English languages that are far more
        affected by Unicode normalization.
        """

        token_texts = [
            sanitize_wordpiece(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)
        ]
        token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)
        if self._tokenizer_lowercases:
            text = text.lower()
            token_texts = [t.lower() for t in token_texts]

        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)

            # Did we not find it at all?
            if token_start_index < 0:
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue

            # Did we jump too far?
            non_whitespace_chars_skipped = sum(
                1 for c in text[text_index:token_start_index] if not c.isspace()
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
        return token_offsets

    def _intra_word_tokenize(
        self, string_tokens: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        tokens: List[Token] = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token_string in string_tokens:
            wordpieces = self.tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]

            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend(
                    Token(text=wp_text, text_id=wp_id)
                    for wp_id, wp_text in zip(wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids))
                )
            else:
                offsets.append(None)
        return tokens, offsets

    @staticmethod
    def _increment_offsets(
        offsets: Iterable[Optional[Tuple[int, int]]], increment: int
    ) -> List[Optional[Tuple[int, int]]]:
        return [
            None if offset is None else (offset[0] + increment, offset[1] + increment)
            for offset in offsets
        ]

    def intra_word_tokenize(
        self, string_tokens: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        tokens, offsets = self._intra_word_tokenize(string_tokens)
        tokens = self.add_special_tokens(tokens)
        offsets = self._increment_offsets(offsets, len(self.single_sequence_start_tokens))
        return tokens, offsets

    def intra_word_tokenize_sentence_pair(
        self, string_tokens_a: List[str], string_tokens_b: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpieces[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        tokens_a, offsets_a = self._intra_word_tokenize(string_tokens_a)
        tokens_b, offsets_b = self._intra_word_tokenize(string_tokens_b)
        offsets_b = self._increment_offsets(
            offsets_b,
            (
                len(self.sequence_pair_start_tokens)
                + len(tokens_a)
                + len(self.sequence_pair_mid_tokens)
            ),
        )
        tokens_a = self.add_special_tokens(tokens_a, tokens_b)
        offsets_a = self._increment_offsets(offsets_a, len(self.sequence_pair_start_tokens))

        return tokens_a, offsets_a, offsets_b

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]

        # Make sure we don't change the input parameters
        tokens2 = copy.deepcopy(tokens2)

        # We add special tokens and also set token type ids.
        import dataclasses

        if tokens2 is None:
            return (
                self.single_sequence_start_tokens
                + with_new_type_id(tokens1, self.single_sequence_token_type_id)  # type: ignore
                + self.single_sequence_end_tokens
            )
        else:
            return (
                self.sequence_pair_start_tokens
                + with_new_type_id(tokens1, self.sequence_pair_first_token_type_id)  # type: ignore
                + self.sequence_pair_mid_tokens
                + with_new_type_id(tokens2, self.sequence_pair_second_token_type_id)  # type: ignore
                + self.sequence_pair_end_tokens
            )

    def num_special_tokens_for_sequence(self) -> int:
        return len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)

    def num_special_tokens_for_pair(self) -> int:
        return (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        )
