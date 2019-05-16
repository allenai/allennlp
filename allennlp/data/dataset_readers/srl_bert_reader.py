import logging
from typing import Dict, List

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

START_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

@DatasetReader.register("srl_bert")
class SrlBertReader(SrlReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling using a BERT pretrained model.
    It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The tokens in the sentence. For this reader, these tokens are always
        BERT wordpiece ids.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """
    def __init__(self,
                 bert_model_name: str,
                 lowercase_input: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:

        if token_indexers:
            raise ConfigurationError("The SrlBertReader has a fixed input "
                                     "representation. Do not pass a token_indexer.")
        super().__init__(token_indexers, domain_identifier, lazy)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.lowercase_input = lowercase_input

    def _tokenize_input(self, tokens: List[str]):
        word_piece_tokens: List[str] = []
        offsets = [0]
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            offsets.append(offsets[-1] + len(word_pieces))
            word_piece_tokens.extend(word_pieces)
        del offsets[0]

        wordpieces = [START_TOKEN] + word_piece_tokens + [SEP_TOKEN]

        offsets = [x + 1 for x in offsets]
        return wordpieces, offsets

    @staticmethod
    def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]):
        # account for the fact the offsets are with respect to
        # additional cls token at the start.
        offsets = [x - 1 for x in offsets]
        new_tags = []
        j = 0
        for i, offset in enumerate(offsets):
            tag = tags[i]
            is_o = tag == "O"
            is_start = True
            while j < offset:
                if is_o:
                    new_tags.append("O")

                elif tag.startswith("I"):
                    new_tags.append(tag)

                elif is_start and tag.startswith("B"):
                    new_tags.append(tag)
                    is_start = False

                elif tag.startswith("B"):
                    _, label = tag.split("-", 1)
                    new_tags.append("I-" + label)
                j += 1

        # Add O tags for cls and sep tokens.
        return ["O"] + new_tags + ["O"]

    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        wordpieces, offsets = self._tokenize_input([t.text for t in tokens])

        new_tags = self._convert_tags_to_wordpiece_tags(tags, offsets)
        new_verbs = [1 if  "-V" in tag else 0 for tag in new_tags]
        # In order to override the indexing mechanism, we need to set the `text_id`
        # attribute directly. This causes the indexing to use this id.
        token_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                token_indexers=self._token_indexers)

        fields: Dict[str, Field] = {}
        fields["tokens"] = token_field
        # pylint: disable=arguments-differ
        fields['verb_indicator'] = SequenceLabelField(new_verbs, token_field)
        if tags:
            fields['tags'] = SequenceLabelField(new_tags, token_field)

        if all([x == 0 for x in new_verbs]):
            verb = None
        else:
            verb = verb_label.index(1)
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
                                            "verb": verb,
                                            "tags": tags,
                                            "offsets": offsets})
        return Instance(fields)
