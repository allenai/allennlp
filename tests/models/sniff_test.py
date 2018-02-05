# pylint: disable=no-self-use,line-too-long

from allennlp.commands.serve import DEFAULT_MODELS
from allennlp.common.testing import AllenNlpTestCase


class SniffTest(AllenNlpTestCase):

    def test_config(self):
        assert set(DEFAULT_MODELS.keys()) == {
                'machine-comprehension',
                'semantic-role-labeling',
                'textual-entailment',
                'coreference-resolution',
                'named-entity-recognition',
        }


    def test_machine_comprehension(self):
        predictor = DEFAULT_MODELS['machine-comprehension'].predictor()

        passage = """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """  # pylint: disable=line-too-long
        question = "Who stars in The Matrix?"

        result = predictor.predict_json({"passage": passage, "question": question})

        correct = "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"

        assert correct == result["best_span_str"]


    def test_semantic_role_labeling(self):
        predictor = DEFAULT_MODELS['semantic-role-labeling'].predictor()

        sentence = "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({"sentence": sentence})

        assert result["tokens"] == [
                "If", "you", "liked", "the", "music", "we", "were", "playing", "last", "night", ",",
                "you", "will", "absolutely", "love", "what", "we", "'re", "playing", "tomorrow", "!"
        ]

        assert result["words"] == [
                "If", "you", "liked", "the", "music", "we", "were", "playing", "last", "night", ",",
                "you", "will", "absolutely", "love", "what", "we", "'re", "playing", "tomorrow", "!"
        ]

        assert result["verbs"] == [
                {"verb": "liked",
                 "description": "If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !",
                 "tags": ["O", "B-ARG0", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
                {"verb": "were",
                 "description": "If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !",
                 "tags": ["O", "O", "O", "O", "O", "O", "B-V", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
                {"verb": "playing",
                 "description": "If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last night] , you will absolutely love what we 're playing tomorrow !",
                 "tags": ["O", "O", "O", "B-ARG1", "I-ARG1", "B-ARG0", "O", "B-V", "B-ARGM-TMP", "I-ARGM-TMP", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
                {"verb": "will",
                 "description": "[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [V: will] [ARG1: absolutely love what we 're playing tomorrow] !",
                 "tags": ["B-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "O", "B-ARG0", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O"]},
                {"verb": "love",
                 "description": "[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !",
                 "tags": ["B-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "I-ARGM-ADV", "O", "B-ARG0", "B-ARGM-MOD", "B-ARGM-ADV", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O"]},
                {"verb": "'re",
                 "description": "If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !",
                 "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-V", "O", "O", "O"]},
                {"verb": "playing",
                 "description": "If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !",
                 "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ARG1", "B-ARG0", "O", "B-V", "B-ARGM-TMP", "O"]}
        ]

    def test_textual_entailment(self):
        predictor = DEFAULT_MODELS['textual-entailment'].predictor()

        result = predictor.predict_json({
                "premise": "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
                "hypothesis": "The spacecraft has the ability to travel between planets."
        })

        assert result["label_probs"][0] > 0.7  # entailment

        result = predictor.predict_json({
                "premise": "Two women are wandering along the shore drinking iced tea.",
                "hypothesis": "Two women are sitting on a blanket near some rocks talking about politics."
        })

        assert result["label_probs"][1] > 0.8  # contradiction

        result = predictor.predict_json({
                "premise": "A large, gray elephant walked beside a herd of zebras.",
                "hypothesis": "The elephant was lost."
        })

        assert result["label_probs"][2] > 0.7  # neutral

    def test_coreference_resolution(self):
        predictor = DEFAULT_MODELS['coreference-resolution'].predictor()

        document = "We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes ."

        result = predictor.predict_json({"document": document})
        print(result)
        assert result['clusters'] == [[[0, 0], [10, 10]],
                                      [[33, 33], [37, 37]],
                                      [[26, 27], [42, 43]]]
        assert result["document"] == ['We', "'re", 'not', 'going', 'to', 'skimp', 'on', 'quality', ',', 'but', 'we', 'are',
                                      'very', 'focused', 'to', 'make', 'next', 'year', '.', 'The', 'only', 'problem', 'is',
                                      'that', 'some', 'of', 'the', 'fabrics', 'are', 'wearing', 'out', '-', 'since', 'I', 'was',
                                      'a', 'newbie', 'I', 'skimped', 'on', 'some', 'of', 'the', 'fabric', 'and', 'the', 'poor',
                                      'quality', 'ones', 'are', 'developing', 'holes', '.']


    def test_ner(self):
        predictor = DEFAULT_MODELS['named-entity-recognition'].predictor()

        sentence = """Michael Jordan is a professor at Berkeley."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == ["Michael", "Jordan", "is", "a", "professor", "at", "Berkeley", "."]
        assert result["tags"] == ["B-PER", "L-PER", "O", "O", "O", "O", "U-LOC", "O"]
