from typing import List
from overrides import overrides

import spacy

from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model


class SentenceSplitter(Registrable):
    """
    A ``SentenceSplitter`` splits strings into sentences.
    """
    default_implementation = 'spacy'

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits ``texts`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        """
        This method lets you take advantage of spacy's batch processing.
        Default implementation is to just iterate over the texts and call ``split_sentences``.
        """
        return [self.split_sentences(text) for text in texts]


@SentenceSplitter.register('spacy')
class SpacySentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's built-in sentence boundary detection.

    Spacy's default sentence splitter uses a dependency parse to detect sentence boundaries, so
    it is slow, but accurate.

    Another option is to use rule-based sentence boundary detection. It's fast and has a small memory footprint,
    since it uses punctuation to detect sentence boundaries. This can be activated with the `rule_based` flag.

    By default, ``SpacySentenceSplitter`` calls the default spacy boundary detector.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 rule_based: bool = False) -> None:
        # we need spacy's dependency parser if we're not using rule-based sentence boundary detection.
        self.spacy = get_spacy_model(language, parse=not rule_based, ner=False, pos_tags=False)
        if rule_based:
            # we use `sentencizer`, a built-in spacy module for rule-based sentence boundary detection.
            # depending on the spacy version, it could be called 'sentencizer' or 'sbd'
            sbd_name = 'sbd' if spacy.__version__ < '2.1' else 'sentencizer'
            if not self.spacy.has_pipe(sbd_name):
                sbd = self.spacy.create_pipe(sbd_name)
                self.spacy.add_pipe(sbd)

    @overrides
    def split_sentences(self, text: str) -> List[str]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        return [[sentence.string.strip() for sentence in doc.sents] for doc in self.spacy.pipe(texts)]
