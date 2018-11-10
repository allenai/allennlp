# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.sentence_splitter import (SpacyRuleBasedSentenceSplitter,
                                                        SpacyStatisticalSentenceSplitter)


class TestSentenceSplitter(AllenNlpTestCase):

    def test_sentence_passes_through_correctly(self):
        splitter = SpacyRuleBasedSentenceSplitter()
        text = ("This is the first sentence. This is the second sentence! "
                "Here's the '3rd' sentence - cool. And this is a fourth sentence.")
        tokens = splitter.split_sentences(text)
        expected_tokens = ["This is the first sentence.", "This is the second sentence!",
                           "Here's the '3rd' sentence - cool.", "And this is a fourth sentence."]
        assert tokens == expected_tokens

    def test_batch_rule_based_sentence_splitting(self):
        splitter = SpacyRuleBasedSentenceSplitter()
        text = ["This is a sentence. This is a second subsentence.",
                "This isn't a sentence. This is a second subsentence! This is a third subsentence.",
                "This is the 3rd sentence?"
                "Here's the 'fourth' sentence - cool. And this is a second subsentence."]
        batch_split = splitter.batch_split_sentences(text)
        separately_split = [splitter.split_sentences(doc) for doc in text]
        assert len(batch_split) == len(separately_split)
        for batch_doc, separate_doc in zip(batch_split, separately_split):
            assert len(batch_doc) == len(separate_doc)
            for batch_sentence, separate_sentence in zip(batch_doc, separate_doc):
                assert batch_sentence == separate_sentence

    def test_batch_statistical_sentence_splitting(self):
        splitter = SpacyStatisticalSentenceSplitter()
        text = ["This is a sentence. This is a second subsentence.",
                "This isn't a sentence. This is a second subsentence! This is a third subsentence.",
                "This is the 3rd sentence?"
                "Here's the 'fourth' sentence - cool. And this is a second subsentence."]
        batch_split = splitter.batch_split_sentences(text)
        separately_split = [splitter.split_sentences(doc) for doc in text]
        assert len(batch_split) == len(separately_split)
        for batch_doc, separate_doc in zip(batch_split, separately_split):
            assert len(batch_doc) == len(separate_doc)
            for batch_sentence, separate_sentence in zip(batch_doc, separate_doc):
                assert batch_sentence == separate_sentence
