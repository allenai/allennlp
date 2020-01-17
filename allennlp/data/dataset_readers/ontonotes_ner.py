import logging
from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence, to_bioul


logger = logging.getLogger(__name__)


def _normalize_word(word: str):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


@DatasetReader.register("ontonotes_ner")
class OntonotesNamedEntityRecognition(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for fine-grained named entity recognition. It returns a dataset of instances with the
    following fields:

    tokens : `TextField`
        The tokens in the sentence.
    tags : `SequenceLabelField`
        A sequence of BIO tags for the NER classes.

    Note that the "/pt/" directory of the Onotonotes dataset representing annotations
    on the new and old testaments of the Bible are excluded, because they do not contain
    NER annotations.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is `{"tokens": SingleIdTokenIndexer()}`.
    domain_identifier : `str`, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    coding_scheme : `str`, (default = None).
        The coding scheme to use for the NER labels. Valid options are "BIO" or "BIOUL".

    # Returns

    A `Dataset` of `Instances` for Fine-Grained NER.

    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        coding_scheme: str = "BIO",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        if domain_identifier == "pt":
            raise ConfigurationError(
                "The Ontonotes 5.0 dataset does not contain annotations for"
                " the old and new testament sections."
            )
        self._coding_scheme = coding_scheme

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading Fine-Grained NER instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(
                "Filtering to only include file paths containing the %s domain",
                self._domain_identifier,
            )

        for sentence in self._ontonotes_subset(
            ontonotes_reader, file_path, self._domain_identifier
        ):
            tokens = [Token(_normalize_word(t)) for t in sentence.words]
            yield self.text_to_instance(tokens, sentence.named_entities)

    @staticmethod
    def _ontonotes_subset(
        ontonotes_reader: Ontonotes, file_path: str, domain_identifier: str
    ) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if (
                domain_identifier is None or f"/{domain_identifier}/" in conll_file
            ) and "/pt/" not in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[Token],
        ner_tags: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        # Add "tag label" to instance
        if ner_tags is not None:
            if self._coding_scheme == "BIOUL":
                ner_tags = to_bioul(ner_tags, encoding="BIO")
            instance_fields["tags"] = SequenceLabelField(ner_tags, sequence)
        return Instance(instance_fields)
