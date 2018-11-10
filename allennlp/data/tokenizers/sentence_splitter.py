from typing import List
from overrides import overrides
from allennlp.common import Registrable
from allennlp.common.util import get_spacy_model
from allennlp.data.tokenizers.token import Token


class SentenceSplitter(Registrable):
    """
    A ``SentenceSplitter`` splits strings into sentences.  This is typically called a "tokenizer" in NLP,
    because splitting strings into characters is trivial, but we use ``Tokenizer`` to refer to the
    higher-level object that splits strings into tokens (which could just be sentence tokens).
    So, we're using "sentence splitter" here for this.
    """
    default_implementation = 'spacy_rulebased'

    def batch_split_sentences(self, texts: List[str]) -> List[List[Token]]:
        """
        Spacy needs to do batch processing, or it can be really slow.  This method lets you take
        advantage of that if you want.  Default implementation is to just iterate of the sentences
        and call ``split_sentences``, but the ``SpacySentenceSplitter`` will actually do batched
        processing.
        """
        return [self.split_sentences(text) for text in texts]

    def split_sentences(self, text: str) -> List[Token]:
        """
        Splits ``texts`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError

@SentenceSplitter.register('spacy_statistical')
class SpacyStatisticalSentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's statistical sentence boundary detection.
    More accurate but slower, since it uses a dependency parses to detect sentence boundaries.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)

    @overrides
    def split_sentences(self, text: str) -> List[Token]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[Token]]:
        def extract_sentences(doc):
            return [sent.string.strip() for sent in doc.sents]

        return [extract_sentences(doc)
                for doc in self.spacy.pipe(texts, n_threads=-1)]


@SentenceSplitter.register('spacy_rulebased')
class SpacyRuleBasedSentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's rule-based sentence boundary detection.
    Faster and less memory footprint, since it uses punctuation to detect sentence boundaries.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        if not self.spacy.has_pipe('sbd'):
            sbd = self.spacy.create_pipe('sbd')   # or: nlp.create_pipe('sbd')
            self.spacy.add_pipe(sbd)

    @overrides
    def split_sentences(self, text: str) -> List[Token]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[Token]]:
        def extract_sentences(doc):
            return [sent.string.strip() for sent in doc.sents]

        return [extract_sentences(doc)
                for doc in self.spacy.pipe(texts, n_threads=-1)]
