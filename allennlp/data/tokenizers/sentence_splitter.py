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
    default_implementation = 'spacy'

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


@SentenceSplitter.register('spacy')
class SpacySentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's tokenizer.
    """
    def __init__(self,
                 language: str = 'en_core_web_sm',
                 pos_tags: bool = False,
                 parse: bool = False,
                 ner: bool = False) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        self.spacy.add_pipe(self.spacy.create_pipe('sentencizer'))

    @overrides
    def split_sentences(self, text: str) -> List[Token]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[Token]]:
        return [sent.string.strip()
                for sent in self.spacy.pipe(texts, n_threads=-1).sents]
