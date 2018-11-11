from typing import List
from overrides import overrides
from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token


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
                 rule_based: bool = False,
                 pos_tags: bool = False,
                 ner: bool = False) -> None:
        # we need spacy's dependency parser if we're not using rule-based sentence boundary detection.
        use_parse = not rule_based
        self.spacy = get_spacy_model(language, pos_tags=pos_tags, parse=use_parse, ner=ner)
        if rule_based:
            # we use `sbd`, a built-in spacy module for rule-based sentence boundary detection. 
            if not self.spacy.has_pipe('sbd'):
                sbd = self.spacy.create_pipe('sbd')
                self.spacy.add_pipe(sbd)

    @overrides
    def split_sentences(self, text: str) -> List[str]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        def extract_sentences(doc):
            return [sent.string.strip() for sent in doc.sents]

        return [extract_sentences(doc)
                for doc in self.spacy.pipe(texts)]
