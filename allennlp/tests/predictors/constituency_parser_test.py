# pylint: disable=no-self-use,invalid-name,protected-access
from nltk import Tree

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.predictors.constituency_parser import LINK_TO_LABEL, NODE_TYPE_TO_STYLE


class TestConstituencyParserPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "What a great test sentence.",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'constituency_parser' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'constituency-parser')

        result = predictor.predict_json(inputs)

        assert len(result["spans"]) == 21 # number of possible substrings of the sentence.
        assert len(result["class_probabilities"]) == 21
        assert result["tokens"] == ["What", "a", "great", "test", "sentence", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

    def test_batch_prediction(self):
        inputs = [
                {"sentence": "What a great test sentence."},
                {"sentence": "Here's another good, interesting one."}
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'constituency_parser' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'constituency-parser')
        results = predictor.predict_batch_json(inputs)

        result = results[0]
        assert len(result["spans"]) == 21 # number of possible substrings of the sentence.
        assert len(result["class_probabilities"]) == 21
        assert result["tokens"] == ["What", "a", "great", "test", "sentence", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

        result = results[1]

        assert len(result["spans"]) == 36 # number of possible substrings of the sentence.
        assert len(result["class_probabilities"]) == 36
        assert result["tokens"] == ["Here", "'s", "another", "good", ",", "interesting", "one", "."]
        assert isinstance(result["trees"], str)

        for class_distribution in result["class_probabilities"]:
            self.assertAlmostEqual(sum(class_distribution), 1.0, places=4)

    def test_build_hierplane_tree(self):
        tree = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
        archive = load_archive(self.FIXTURES_ROOT / 'constituency_parser' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'constituency-parser')

        hierplane_tree = predictor._build_hierplane_tree(tree, 0, is_root=True)

        # pylint: disable=bad-continuation
        correct_tree = {
                'text': 'the dog chased the cat',
                "linkNameToLabel": LINK_TO_LABEL,
                "nodeTypeToStyle": NODE_TYPE_TO_STYLE,
                'root': {
                        'word': 'the dog chased the cat',
                        'nodeType': 'S',
                        'attributes': ['S'],
                        'link': 'S',
                        'children': [{
                                'word': 'the dog',
                                'nodeType': 'NP',
                                'attributes': ['NP'],
                                'link': 'NP',
                                'children': [{
                                        'word': 'the',
                                        'nodeType': 'D',
                                        'attributes': ['D'],
                                        'link': 'D'
                                        },
                                        {
                                        'word': 'dog',
                                        'nodeType': 'N',
                                        'attributes': ['N'],
                                        'link': 'N'}
                                        ]
                                },
                                {
                                'word': 'chased the cat',
                                'nodeType': 'VP',
                                'attributes': ['VP'],
                                'link': 'VP',
                                'children': [{
                                    'word': 'chased',
                                    'nodeType': 'V',
                                    'attributes': ['V'],
                                    'link': 'V'
                                    },
                                    {
                                    'word':
                                    'the cat',
                                    'nodeType': 'NP',
                                    'attributes': ['NP'],
                                    'link': 'NP',
                                    'children': [{
                                            'word': 'the',
                                            'nodeType': 'D',
                                            'attributes': ['D'],
                                            'link': 'D'
                                            },
                                            {
                                            'word': 'cat',
                                            'nodeType': 'N',
                                            'attributes': ['N'],
                                            'link': 'N'}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
        # pylint: enable=bad-continuation
        assert correct_tree == hierplane_tree
