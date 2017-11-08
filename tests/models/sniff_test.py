# pylint: disable=line-too-long,bad-whitespace,no-self-use

from allennlp.commands import DEFAULT_MODELS
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class SniffTest(AllenNlpTestCase):

    def test_config(self):
        assert set(DEFAULT_MODELS.keys()) == {
                'machine-comprehension',
                'semantic-role-labeling',
                'textual-entailment'
        }


    def test_machine_comprehension(self):
        predictor = Predictor.from_archive(
                load_archive(DEFAULT_MODELS['machine-comprehension']),
                'machine-comprehension'
        )

        passage = """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """
        question = "Who stars in The Matrix?"

        result = predictor.predict_json({"passage": passage, "question": question})

        correct = "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"

        assert correct == result["best_span_str"]


    def test_semantic_role_labeling(self):
        predictor = Predictor.from_archive(
                load_archive(DEFAULT_MODELS['semantic-role-labeling']),
                'semantic-role-labeling'
        )

        sentence = "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({"sentence": sentence})

        assert result["tokens"] == ["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"]

        assert result["words"] == ["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"]

        assert result["verbs"] == [
                {"verb":"liked","description":"If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O","O","O","O","O","O","O","O","O","O","O"]},
                {"verb":"were","description":"If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","O","O","O","B-V","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},
                {"verb":"playing","description":"If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","B-ARG1","I-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","I-ARGM-TMP","O","O","O","O","O","O","O","O","O","O","O"]},
                {"verb":"will","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [V: will] [ARG1: absolutely love what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},
                {"verb":"love","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-ARGM-MOD","B-ARGM-ADV","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},
                {"verb":"'re","description":"If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-V","O","O","O"]},
                {"verb":"playing","description":"If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","O"]}
        ]

    def test_textual_entailment(self):
        predictor = Predictor.from_archive(
                load_archive(DEFAULT_MODELS['textual-entailment']),
                'textual-entailment'
        )

        result = predictor.predict_json({
                "premise": "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
                "hypothesis":"The spacecraft has the ability to travel between planets."
        })

        assert result["label_probs"][0] > 0.7  # entailment

        result = predictor.predict_json({
                "premise":"Two women are wandering along the shore drinking iced tea.",
                "hypothesis":"Two women are sitting on a blanket near some rocks talking about politics."
        })

        assert result["label_probs"][1] > 0.8  # contradiction

        result = predictor.predict_json({
                "premise":"A large, gray elephant walked beside a herd of zebras.",
                "hypothesis":"The elephant was lost."
        })

        assert result["label_probs"][2] > 0.7  # neutral
