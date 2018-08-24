# pylint: disable=no-self-use,line-too-long

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

def demo_model(archive_file: str, predictor_name: str) -> Predictor:
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)

# TODO(joelgrus): this is duplicated in the demo repo
# figure out where it really belongs
DEFAULT_MODELS = {
        'machine-comprehension': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz',  # pylint: disable=line-too-long
                'machine-comprehension'
        ),
        'semantic-role-labeling': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz', # pylint: disable=line-too-long
                'semantic-role-labeling'
        ),
        'textual-entailment': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz',  # pylint: disable=line-too-long
                'textual-entailment'
        ),
        'coreference-resolution': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',  # pylint: disable=line-too-long
                'coreference-resolution'
        ),
        'named-entity-recognition': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz',  # pylint: disable=line-too-long
                'sentence-tagger'
        ),
        'constituency-parsing': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',  # pylint: disable=line-too-long
                'constituency-parser'
        ),
        'dependency-parsing': (
                'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz',  # pylint: disable=line-too-long
                'biaffine-dependency-parser'
        )
}


class SniffTest(AllenNlpTestCase):

    def test_machine_comprehension(self):
        predictor = demo_model(*DEFAULT_MODELS['machine-comprehension'])

        passage = """The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """  # pylint: disable=line-too-long
        question = "Who stars in The Matrix?"

        result = predictor.predict_json({"passage": passage, "question": question})

        correct = "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"

        assert correct == result["best_span_str"]

    def test_semantic_role_labeling(self):
        predictor = demo_model(*DEFAULT_MODELS['semantic-role-labeling'])

        sentence = "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({"sentence": sentence})

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
                 "description": "If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last] night , you will absolutely love what we 're playing tomorrow !",
                 "tags": ["O", "O", "O", "B-ARG1", "I-ARG1", "B-ARG0", "O", "B-V", "B-ARGM-TMP", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
                {"verb": "will",
                 "description": "If you liked the music we were playing last night , you [V: will] absolutely love what we 're playing tomorrow !",
                 "tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-V", "O", "O", "O", "O", "O", "O", "O", "O"]},
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
        predictor = demo_model(*DEFAULT_MODELS['textual-entailment'])

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

        assert result["label_probs"][2] > 0.6  # neutral

    def test_coreference_resolution(self):
        predictor = demo_model(*DEFAULT_MODELS['coreference-resolution'])

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
        predictor = demo_model(*DEFAULT_MODELS['named-entity-recognition'])

        sentence = """Michael Jordan is a professor at Berkeley."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["words"] == ["Michael", "Jordan", "is", "a", "professor", "at", "Berkeley", "."]
        assert result["tags"] == ["B-PER", "L-PER", "O", "O", "O", "O", "U-LOC", "O"]

    def test_constituency_parsing(self):
        predictor = demo_model(*DEFAULT_MODELS['constituency-parsing'])

        sentence = """Pierre Vinken died aged 81; immortalised aged 61."""

        result = predictor.predict_json({"sentence": sentence})

        assert result["tokens"] == ["Pierre", "Vinken", "died", "aged", "81", ";", "immortalised", "aged", "61", "."]
        assert result["trees"] == "(S (NP (NNP Pierre) (NNP Vinken)) (VP (VP (VBD died) (NP (JJ aged) (CD 81))) (, ;) (VP (VBD immortalised) (S (ADJP (VBN aged) (NP (CD 61)))))) (. .))"

    def test_dependency_parsing(self):
        predictor = demo_model(*DEFAULT_MODELS['dependency-parsing'])
        sentence = """He ate spaghetti with chopsticks."""
        result = predictor.predict_json({"sentence": sentence})
        # Note that this tree is incorrect. We are checking here that the decoded
        # tree is _actually a tree_ - in greedy decoding versions of the dependency
        # parser, this sentence has multiple heads. This test shouldn't really live here,
        # but it's very difficult to re-create a concrete example of this behaviour without
        # a trained dependency parser.
        assert result['words'] == ['He', 'ate', 'spaghetti', 'with', 'chopsticks', '.']
        assert result['pos'] == ['PRP', 'VBD', 'NNS', 'IN', 'NNS', '.']
        assert result['predicted_dependencies'] == ['nsubj', 'root', 'dobj', 'prep', 'pobj', 'punct']
        assert result['predicted_heads'] == [2, 0, 2, 2, 4, 2]
