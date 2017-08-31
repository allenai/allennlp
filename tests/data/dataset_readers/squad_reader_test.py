# pylint: disable=no-self-use,invalid-name
from allennlp.common import Params
from allennlp.data.dataset_readers import SquadReader
from allennlp.data.dataset_readers.squad import _char_span_to_token_span
from allennlp.data.tokenizers import WordTokenizer
from allennlp.common.testing import AllenNlpTestCase


class TestSquadReader(AllenNlpTestCase):
    def test_char_span_to_token_span_handles_easy_cases(self):
        # These are _inclusive_ spans, on both sides.
        tokenizer = WordTokenizer()
        passage = "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy " +\
            "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four " +\
            "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her " +\
            "first performances since giving birth to Blue Ivy."
        _, offsets = tokenizer.tokenize(passage)
        # "January 7, 2012"
        token_span = _char_span_to_token_span(offsets, (3, 18))[0]
        assert token_span == (1, 4)
        # "Lenox Hill Hospital"
        token_span = _char_span_to_token_span(offsets, (91, 110))[0]
        assert token_span == (22, 24)
        # "Lenox Hill Hospital in New York."
        token_span = _char_span_to_token_span(offsets, (91, 123))[0]
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
        _, offsets = tokenizer.tokenize(passage)
        token_span = _char_span_to_token_span(offsets, (start, end))[0]
        assert token_span == (184, 185)

    def test_read_from_file(self):
        reader = SquadReader()
        instances = reader.read('tests/fixtures/data/squad.json').instances
        assert len(instances) == 5

        assert instances[0].fields["question"].tokens[:3] == ["To", "whom", "did"]
        assert instances[0].fields["passage"].tokens[:3] == ["Architecturally", ",", "the"]
        assert instances[0].fields["passage"].tokens[-3:] == ["of", "Mary", "."]
        assert instances[0].fields["span_start"].sequence_index == 102
        assert instances[0].fields["span_end"].sequence_index == 104

        assert instances[1].fields["question"].tokens[:3] == ["What", "sits", "on"]
        assert instances[1].fields["passage"].tokens[:3] == ["Architecturally", ",", "the"]
        assert instances[1].fields["passage"].tokens[-3:] == ["of", "Mary", "."]
        assert instances[1].fields["span_start"].sequence_index == 17
        assert instances[1].fields["span_end"].sequence_index == 23

        # We're checking this case because I changed the answer text to only have a partial
        # annotation for the last token, which happens occasionally in the training data.  We're
        # making sure we get a reasonable output in that case here.
        assert instances[3].fields["question"].tokens[:3] == ["Which", "individual", "worked"]
        assert instances[3].fields["passage"].tokens[:3] == ["In", "1882", ","]
        assert instances[3].fields["passage"].tokens[-3:] == ["Nuclear", "Astrophysics", "."]
        span_start = instances[3].fields["span_start"].sequence_index
        span_end = instances[3].fields["span_end"].sequence_index
        answer_tokens = instances[3].fields["passage"].tokens[span_start:(span_end + 1)]
        expected_answer_tokens = ["Father", "Julius", "Nieuwland"]
        assert answer_tokens == expected_answer_tokens

    def test_can_build_from_params(self):
        reader = SquadReader.from_params(Params({}))
        # pylint: disable=protected-access
        assert reader._tokenizer.__class__.__name__ == 'WordTokenizer'
        assert reader._token_indexers["tokens"].__class__.__name__ == 'SingleIdTokenIndexer'
