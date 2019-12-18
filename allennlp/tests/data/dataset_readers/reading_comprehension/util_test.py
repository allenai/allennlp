from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.tokenizers import SpacyTokenizer


class TestReadingComprehensionUtil(AllenNlpTestCase):
    def test_char_span_to_token_span_handles_easy_cases(self):
        # These are _inclusive_ spans, on both sides.
        tokenizer = SpacyTokenizer()
        passage = (
            "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy "
            + "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four "
            + "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her "
            + "first performances since giving birth to Blue Ivy."
        )
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

    def test_char_span_to_token_span_handles_last_token(self):
        # An even earlier version of the code had a hard time when the answer was the last token in
        # the passage.  This tests that case, on the instance that used to fail.
        tokenizer = SpacyTokenizer()
        passage = (
            "Beyonc\u00e9 is believed to have first started a relationship with Jay Z "
            + 'after a collaboration on "\'03 Bonnie & Clyde", which appeared on his seventh '
            + "album The Blueprint 2: The Gift & The Curse (2002). Beyonc\u00e9 appeared as Jay "
            + "Z's girlfriend in the music video for the song, which would further fuel "
            + "speculation of their relationship. On April 4, 2008, Beyonc\u00e9 and Jay Z were "
            + "married without publicity. As of April 2014, the couple have sold a combined 300 "
            + "million records together. The couple are known for their private relationship, "
            + "although they have appeared to become more relaxed in recent years. Beyonc\u00e9 "
            + 'suffered a miscarriage in 2010 or 2011, describing it as "the saddest thing" '
            + "she had ever endured. She returned to the studio and wrote music in order to cope "
            + "with the loss. In April 2011, Beyonc\u00e9 and Jay Z traveled to Paris in order "
            + "to shoot the album cover for her 4, and unexpectedly became pregnant in Paris."
        )
        start = 912
        end = 912 + len("Paris.")
        tokens = tokenizer.tokenize(passage)
        offsets = [(t.idx, t.idx + len(t.text)) for t in tokens]
        token_span = util.char_span_to_token_span(offsets, (start, end))[0]
        assert token_span == (184, 185)

    def test_char_span_to_token_span_handles_undertokenization(self):
        tokenizer = SpacyTokenizer()
        passage = "This sentence will have two under tokenized tokens, one#here and one at the#end"
        tokens = tokenizer.tokenize(passage)
        offsets = [(t.idx, t.idx + len(t.text)) for t in tokens]

        # scenario 1: under tokenized in the middle of the sentence, look for the first part of the token
        start = 52
        end = start + len("one")
        expected_span = (9, 9)  # the indices of the whole "one&here" token should be returned
        token_span, error = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error

        # scenario 2: under tokenized in the middle of the sentence, look for the second part of the token
        start = 56
        end = start + len("here")
        expected_span = (9, 9)  # the indices of the whole "one&here" token should be returned
        token_span, error = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error

        # scenario 3: under tokenized at the end of the sentence, look for the first part of the token
        start = 72
        end = start + len("the")
        expected_span = (13, 13)  # the indices of the whole "the&end" token should be returned
        token_span, error = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error

        # scenario 4: under tokenized at the end of the sentence, look for the second part of the token
        # this used to cause an IndexError
        start = 76
        end = start + len("end")
        expected_span = (13, 13)  # the indices of the whole "the&end" token should be returned
        token_span, errory = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error

    def test_char_span_to_token_span_handles_out_of_bounds_start_end(self):
        tokenizer = SpacyTokenizer()
        passage = "This sentence is just for testing purposes"
        tokens = tokenizer.tokenize(passage)
        offsets = [(t.idx, t.idx + len(t.text)) for t in tokens]

        # scenario 1: negative start character span (this should really never happen)
        start = -1
        end = start + len("This")
        expected_span = (0, 0)
        token_span, error = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error

        # scenario 2: end character span exceeds sentence length, for whichever reason
        start = 34
        end = start + len("purposes") + 1
        expected_span = (6, 6)
        token_span, error = util.char_span_to_token_span(offsets, (start, end))
        assert token_span == expected_span
        assert error
