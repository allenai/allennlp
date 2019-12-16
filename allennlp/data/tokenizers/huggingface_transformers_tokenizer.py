from typing import Dict, List, Any
from overrides import overrides
import logging

import sys
import unicodedata
import html

from allennlp.data.tokenizers import Token, Tokenizer

from transformers.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)

CONTROL_CHARS = u"".join(
    chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c))[0] == "C"
)
SPIECE_UNDERLINE = u"▁"


@Tokenizer.register("huggingface_transformers")
class HuggingfaceTransformersTokenizer(Tokenizer):
    """
    A ``HuggingfaceTransformersTokenizer`` uses a model from HuggingFace's
    ``transformers`` library to tokenize some input text.  This often means wordpieces
    (where ``'AllenNLP is awesome'`` might get split into ``['Allen', '##NL', '##P', 'is',
    'awesome']``), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.

    We take a model name as an input parameter, which we will pass to
    ``AutoTokenizer.from_pretrained``.

    A suggestion for a usage *without* TokenIndexer, in order to fully utilize the Hugging Face's API:
    1. Tokenize with ``tokenize`` or ``tokenize_with_offsets``
    2. Encode the tokens with ``encode_plus``, in order to get the appropriate form the pretrained model expects
    3. Create a LabelsField field with the encoded tokens
    4. Optioanlly, add fields such as `token_type_ids` and `special_tokens_mask`

    Example:
        tokens = tokenizer.tokenize(text)
        encoded_inputs = self._tokenizer.encode_plus([token.text for token in tokens],
                                    add_special_tokens=True,
                                    return_token_type_ids=True,
                                    return_special_tokens_mask=True)
        fields: Dict[str, Field] = {}
        fields['tokens'] = LabelsField(encoded_inputs['input_ids'])
        fields['token_type_ids'] = LabelsField(encoded_inputs['token_type_ids'])
        fields['special_tokens_mask'] = LabelsField(encoded_inputs['special_tokens_mask'])
        fields['pad_mask'] = LabelsField([1] * len(question_passage_tokens)) # For right padding

    pretrained_model : ``str``
        The name of the tokenizer to use.
    init_kwargs : ``Dict[str, Any]
        Arguments to pass to the constructor of the tokenizer
    kwargs : ``Dict[str, Any]
        Arguments to pass to the ``tokenize`` method
    """

    def __init__(
        self, pretrained_model: str, init_kwargs: Dict[str, Any] = {}, kwargs: Dict[str, Any] = {}
    ):
        self._pretrained_model = pretrained_model
        self._kwargs = kwargs
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **init_kwargs)
        self._tokenization_cache: Dict[str, str] = {}

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text=token) for token in self._tokenizer.tokenize(text, **self._kwargs)]

    def tokenize_with_offsets(self, text: str) -> List[Token]:
        tokens, offsets = self._tokenize_with_offsets(text, **self._kwargs)

        tokens_with_offsets = []
        for i, offset in enumerate(offsets):
            if i < len(offsets) - 1:
                next_offset = offsets[i + 1]
                original_token_text = text[offset:next_offset]
            else:
                original_token_text = text[offset:]
            original_token_text = original_token_text.strip()
            if not original_token_text:
                original_token_text = " "
            tokens_with_offsets.append(
                Token(text=tokens[i], idx=offset, original_text=original_token_text)
            )

        return tokens_with_offsets

    @property
    def encode_plus(self):
        return self._tokenizer.encode_plus

    def _detokenize_for_offsets(self, tok):
        """
        Remove any tokenization artifacts for sub-word tokens.
        Used by tokenize_with_offsets to match tokens back in the original text.
         (like ## prefix for BERT)
        :param tok: the token from the tokenizer
        :return: the token as it would have looked (best approximation) in the original text
        """
        if "t5" in self._pretrained_model:
            return self._tokenizer.sp_model.decode_pieces([tok])
        elif "distilbert" in self._pretrained_model:
            if tok.startswith("##"):
                return tok[2:]
            return tok.strip()
        elif "albert" in self._pretrained_model:
            return tok.replace(SPIECE_UNDERLINE, " ").strip()
        elif "roberta" in self._pretrained_model:
            return (
                bytearray([self.byte_decoder[c] for c in tok])
                .decode("utf-8", errors=self.errors)
                .strip()
            )
        elif "bert-base-japanese" in self._pretrained_model:
            if tok.startswith("##"):
                return tok[2:]
            return tok.strip()
        elif "bert" in self._pretrained_model:
            if tok.startswith("##"):
                return tok[2:]
            return tok.strip()
        elif "openai-gpt" in self._pretrained_model:
            return tok.replace("</w>", " ").strip()
        elif "gpt2" in self._pretrained_model:
            return (
                bytearray([self.byte_decoder[c] for c in tok])
                .decode("utf-8", errors=self.errors)
                .strip()
            )
        elif "xlnet" in self._pretrained_model:
            return tok.replace(SPIECE_UNDERLINE, " ").strip()
        elif "xlm" in self._pretrained_model:
            return tok.replace("</w>", " ").strip()
        elif "ctrl" in self._pretrained_model:
            if tok.endswith("@@"):
                return tok[:-2]
            return tok.strip()
        return tok.strip()

    def _tokenize_with_offsets(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer
            and also keeps track of the token offsets relative to the original text.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.

            Keep track of token offsets by trying to progressively tokenize the text character by character,
            and consume matching tokens along the way.
            For efficiency, parts of the text that were already matched should be discarded.
            However, since some tokenizations are dependent on neighboring past text,
            in cases of mismatch we fall back to use previously matched parts of the text.
            For some tokenizers, more than a single token can be generated for a single character.
            In such cases, the nonfinal tokens will have the same offset as the final token.

            Tested with the following:
            - OpenAIGPTTokenizer
            - TransfoXLTokenizer
            - DistilBertTokenizer
            - BertTokenizer
            - BertJapaneseTokenizer
            - RobertaTokenizer
            - GPT2Tokenizer
            - XLNetTokenizer
            - AlbertTokenizer
            - CTRLTokenizer
            - XLMTokenizer - Problem in the tokenization, specifics are in the comments
            - T5Tokenizer

            Args:
                text: The sequence to be encoded.
                **kwargs: passed to the child `self.tokenize()` method

            Return:
                A tuple of shape::
                    (tokens: list[str], offsets: list[int])
            With the fields:
                ``tokens``: list of tokens
                ``offsets``: list of offsets in the text corresponsing to ``tokens``

        """

        def get_max_space_length(text):
            original_text = text
            text = unescape_html_and_remove_control_chars(text)
            max_space_length = 0
            count = False
            for i, c in enumerate(text):
                if c == " ":
                    if not count:
                        count = True
                        start_index = i
                else:
                    if count:
                        count = False
                        max_space_length = max(max_space_length, i - start_index)
            # with a safety margin
            return max_space_length + (len(original_text) - len(text))

        def is_prefix(lst, other_lst):
            """
            Checks if `lst` is a prefix of `other_lst`
            """
            if len(lst) > len(other_lst):
                return False
            return other_lst[: len(lst)] == lst

        def get_comparison_tokens(
            tokenize_func,
            text,
            boundary_token_offset,
            token_offset,
            search_length,
            cache,
            **tokenize_func_kwargs
        ):
            search_text = text[boundary_token_offset : token_offset + search_length]
            if search_text not in cache:
                cache[search_text] = tokenize_func(search_text, **tokenize_func_kwargs)
            return cache[search_text]

        def remove_control_chars(text):
            for control_char in CONTROL_CHARS:
                text = text.replace(control_char, "")
            return text

        def unescape_html_and_remove_control_chars(text):
            text = html.unescape(text)
            return remove_control_chars(text)

        # Tokenize text
        tokens = self._tokenizer.tokenize(text, **kwargs)
        is_lower_casing = (
            True  # TODO: Recommended way to handle this? Keeping it always True would also be okay
        )

        # Get maximum search length
        max_word_length = max([len(word) for word in text.split()])
        max_space_length = get_max_space_length(text)
        max_search_length = max_word_length + max_space_length

        # Initialize token iteration variables
        boundary_token_index = 0
        prev_boundary_token_indexes = [0]
        offsets = [0]
        i = 0
        retry = 0
        while True:
            token = tokens[i]
            match_error = False

            if retry > 0:
                # The tokenization from the boundary doesn't match the text, retrying with a previous boundary
                boundary_token_index = prev_boundary_token_indexes[-retry]
            else:
                # Try boundary of the current token
                boundary_token_index = i

            # Initialize search variables
            offset = offsets[i]
            search_length = 1
            comparison_tokens = []
            while True:
                comparison_tokens = get_comparison_tokens(
                    self._tokenizer.tokenize,
                    text,
                    offsets[boundary_token_index],
                    offset,
                    search_length,
                    self._tokenization_cache,
                    **kwargs
                )

                target_tokens = tokens[boundary_token_index : i + 1]
                if is_prefix(target_tokens, comparison_tokens):
                    # Found a tokenization match
                    if len(comparison_tokens) > len(target_tokens):
                        # Handle special cases
                        matching_text = text[offset : offset + search_length]
                        detokenized = self._detokenize_for_offsets(token)
                        if is_lower_casing:
                            matching_text = matching_text.lower()
                            detokenized = detokenized.lower()
                        # TODO: Remove accents with tokenization_xlm.lowercase_and_remove_accent?
                        # Will improve accuracy for XLM
                        index = matching_text.find(detokenized)
                        if index != -1:
                            # Words that have a wordpiece tokenization that
                            # doesn't contain the tokenization of its prefixes.
                            # Example for XLNet:
                            # text = "1603"
                            # tokens = ["▁16", "03"]
                            # tokenization for "160": ["▁160"]
                            search_length = index + len(detokenized)
                        else:
                            # For cases in which the current token won't be produced
                            # without an additional character that is only part of the
                            # text that corresponds to the next tokens.
                            # Example for XLNet:
                            # text = "How many points did the buccaneers need to tie in the first?"
                            # tokens = [..., '▁the', '▁', 'bu', 'cca', 'ne', 'ers', ...]
                            # target_tokens = ['▁']
                            # comparison_tokens = ['▁', 'b']
                            # prev_comparison_tokens = ['']
                            # OR
                            # For characters that are tokenized to two or more tokens
                            search_length = 0

                    # Store successful boundary
                    if prev_boundary_token_indexes[-1] != boundary_token_index:
                        prev_boundary_token_indexes.append(boundary_token_index)
                    retry = 0
                    break

                if search_length == max_search_length:
                    # The tokenization from the boundary doesn't match the text, retry with a previous boundary,
                    # keep retrying until all the previous successful boundaries are used
                    match_error = True
                    if retry < len(prev_boundary_token_indexes):
                        retry += 1
                    else:
                        retry = 0
                    break

                # Search step
                search_length += 1

            if match_error:
                if retry > 0:
                    continue
                # Failed to match offsets to the tokens
                break

            # Keep consuming characters until there's no tokenization match.
            # Required due to special characters such as in
            # "Moskva: Russkiĭ fond sodeĭstviii︠a︡ obrazovanii︠u︡ i nauke"
            if comparison_tokens == target_tokens:
                while True:
                    if len(text) == offset + search_length:
                        break

                    comparison_tokens = get_comparison_tokens(
                        self._tokenizer.tokenize,
                        text,
                        offsets[boundary_token_index],
                        offset,
                        search_length + 1,
                        self._tokenization_cache,
                        **kwargs
                    )
                    if is_prefix(comparison_tokens, target_tokens):
                        search_length += 1
                    else:
                        break

            if len(text) != offset + search_length:
                # Add the next token offset only if the end of the text wasn't reached
                offsets.append(offset + search_length)
            else:
                break
            i += 1

        assert not match_error, "Unknown failure reason"
        # Relaxed for XLM, instead:
        # assert len(tokens) == len(offsets), "Number of tokens doesn't match the number of offsets"
        while len(tokens) != len(offsets):
            offsets.append(len(text) - 1)  # bad, but better than having nothing

        # TODO: Move to a test?
        # Construct the original text that corresponds (up to spaces) to each token
        original_token_texts = []
        for i, offset in enumerate(offsets):
            if i < len(offsets) - 1:
                next_offset = offsets[i + 1]
                original_token_text = text[offset:next_offset]
            else:
                original_token_text = text[offset:]
            original_token_texts.append(original_token_text)
        # We assume whitespaces are a tokenization boundary,
        # so we use this assumption to verify the alignment was valid
        # (considering special cases)
        for i, original_token_text in enumerate(original_token_texts):
            splits = len(original_token_text.strip().split())
            if splits >= 2:
                # Don't warn about cases in which the tokenizer ignores control characters.
                # Example: "a \ufeff test" might be tokenized as ["a", "test"] but matched to ["a \ufeff", "test"]
                original_token_text = remove_control_chars(original_token_text)
                splits = len(original_token_text.strip().split())
            if splits >= 2:
                # Don't warn about cases in which the tokenizer unescapes html entities (OpenAIGPTTokenizer)
                # and ignores control characters.
                # Example: "98&#160; yards" might be tokenized as ["98", "yards"]
                # but matched to ["98", "&#160; yards"]
                original_token_text = unescape_html_and_remove_control_chars(original_token_text)
                splits = len(original_token_text.strip().split())

            if splits == 2:
                # In XLM, "\"." is tokenized to [".", "\""] and similarly "\",",
                # so it's not possible to create contiguous offsets.
                # Also in XLM, "30\xa0000" is tokenized to ['3', '0.', '000</w>'] which is problematic.
                logger.warning(
                    """Possible error:
                                A token is consuming text of another token as well,
                                probably due to a bad character in the input or out-of-order tokenization.
                                Token #%d, %s"""
                    % (i, original_token_text)
                )

            # Not expected to happen
            assert (
                splits <= 2
            ), """Error:
                            A token is consuming text of multiple other tokens as well,
                            probably due to a bad character in the input or out-of-order tokenization.
                            Token #%d, %s""" % (
                i,
                original_token_text,
            )

        return tokens, offsets
