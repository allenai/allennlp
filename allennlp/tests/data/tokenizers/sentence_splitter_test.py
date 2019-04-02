# pylint: disable=no-self-use,invalid-name
import pytest
import spacy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


class TestSentenceSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestSentenceSplitter, self).setUp()
        self.dep_parse_splitter = SpacySentenceSplitter(rule_based=False)
        self.rule_based_splitter = SpacySentenceSplitter(rule_based=True)

    def test_rule_based_splitter_passes_through_correctly(self):
        text = ("This is the first sentence. This is the second sentence! "
                "Here's the '3rd' sentence - yes, it is. And yes; this is a fourth sentence?")
        tokens = self.rule_based_splitter.split_sentences(text)
        expected_tokens = ["This is the first sentence.", "This is the second sentence!",
                           "Here's the '3rd' sentence - yes, it is.", "And yes; this is a fourth sentence?"]
        assert tokens == expected_tokens

    @pytest.mark.skipif(spacy.__version__ < "2.1", reason="this model changed from 2.0 to 2.1")
    def test_dep_parse_splitter_passes_through_correctly(self):
        text = ("This is the first sentence. This is the second sentence! "
                "Here's the '3rd' sentence - yes, it is. And yes; this is a fourth sentence?")
        tokens = self.dep_parse_splitter.split_sentences(text)
        expected_tokens = ["This is the first sentence.", "This is the second sentence!",
                           "Here's the '3rd' sentence - yes, it is.", "And yes; this is a fourth sentence?"]
        assert tokens == expected_tokens

    def test_batch_rule_based_sentence_splitting(self):
        text = ["This is a sentence. This is a second sentence.",
                "This isn't a sentence. This is a second sentence! This is a third sentence.",
                "This is the 3rd sentence?",
                "Here's the 'fourth' sentence - yes, it is. And this is a second sentence."]
        batch_split = self.rule_based_splitter.batch_split_sentences(text)
        separately_split = [self.rule_based_splitter.split_sentences(doc) for doc in text]
        assert len(batch_split) == len(separately_split)
        for batch_doc, separate_doc in zip(batch_split, separately_split):
            assert len(batch_doc) == len(separate_doc)
            for batch_sentence, separate_sentence in zip(batch_doc, separate_doc):
                assert batch_sentence == separate_sentence

    def test_batch_dep_parse_sentence_splitting(self):
        text = ["This is a sentence. This is a second sentence.",
                "This isn't a sentence. This is a second sentence! This is a third sentence.",
                "This is the 3rd sentence?",
                "Here's the 'fourth' sentence - yes, it is. And this is a second sentence."]
        batch_split = self.dep_parse_splitter.batch_split_sentences(text)
        separately_split = [self.dep_parse_splitter.split_sentences(doc) for doc in text]
        assert len(batch_split) == len(separately_split)
        for batch_doc, separate_doc in zip(batch_split, separately_split):
            assert len(batch_doc) == len(separate_doc)
            for batch_sentence, separate_sentence in zip(batch_doc, separate_doc):
                assert batch_sentence == separate_sentence
