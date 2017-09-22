# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.tokenizers import WordTokenizer


class TestReadingComprehensionUtil(AllenNlpTestCase):
    def test_char_span_to_token_span_handles_easy_cases(self):
        # These are _inclusive_ spans, on both sides.
        tokenizer = WordTokenizer()
        passage = "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy " +\
            "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four " +\
            "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her " +\
            "first performances since giving birth to Blue Ivy."
        tokens = tokenizer.tokenize(passage)
        offsets = [(t.idx, t.idx + len(t.text)) for t in tokens]
        # "January 7, 2012"
        token_span = util.char_span_to_token_span(offsets, (3, 18))[0]
        assert token_span == (1, 4)
        # "Lenox Hill Hospital"
        token_span = util.char_span_to_token_span(offsets, (91, 110))[0]
        assert token_span == (22, 24)
        # "Lenox Hill Hospital in New York."
        token_span = util.char_span_to_token_span(offsets, (91, 123))[0]
        assert token_span == (22, 28)

    def test_char_span_to_token_span_handles_hard_cases(self):
        # An earlier version of the code had a hard time when the answer was the last token in the
        # passage.  This tests that case, on the instance that used to fail.
        tokenizer = WordTokenizer()
        passage = "Beyonc\u00e9 is believed to have first started a relationship with Jay Z " +\
            "after a collaboration on \"'03 Bonnie & Clyde\", which appeared on his seventh " +\
            "album The Blueprint 2: The Gift & The Curse (2002). Beyonc\u00e9 appeared as Jay " +\
            "Z's girlfriend in the music video for the song, which would further fuel " +\
            "speculation of their relationship. On April 4, 2008, Beyonc\u00e9 and Jay Z were " +\
            "married without publicity. As of April 2014, the couple have sold a combined 300 " +\
            "million records together. The couple are known for their private relationship, " +\
            "although they have appeared to become more relaxed in recent years. Beyonc\u00e9 " +\
            "suffered a miscarriage in 2010 or 2011, describing it as \"the saddest thing\" " +\
            "she had ever endured. She returned to the studio and wrote music in order to cope " +\
            "with the loss. In April 2011, Beyonc\u00e9 and Jay Z traveled to Paris in order " +\
            "to shoot the album cover for her 4, and unexpectedly became pregnant in Paris."
        start = 912
        end = 912 + len("Paris.")
        tokens = tokenizer.tokenize(passage)
        offsets = [(t.idx, t.idx + len(t.text)) for  t in tokens]
        token_span = util.char_span_to_token_span(offsets, (start, end))[0]
        assert token_span == (184, 185)
