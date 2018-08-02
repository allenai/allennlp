# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestBiaffineDependencyParser(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        inputs = {
                "sentence": "Please could you parse this sentence?",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser'
                               / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

        result = predictor.predict_json(inputs)

        heads = result.get("heads")
        assert heads is not None
        assert isinstance(heads, list)
        assert all(isinstance(x, int) for x in heads)
        head_tags = result.get("head_tags")
        assert head_tags is not None
        assert isinstance(head_tags, list)
        assert all(isinstance(x, int) for x in head_tags)

        predicted_heads = result.get("predicted_heads")
        assert len(predicted_heads) == len(heads) - 1

        predicted_dependencies = result.get("predicted_dependencies")
        assert len(predicted_dependencies) == len(head_tags) - 1
        assert isinstance(predicted_dependencies, list)
        assert all(isinstance(x, str) for x in predicted_dependencies)

        assert result.get("loss") is not None
        assert result.get("arc_loss") is not None
        assert result.get("tag_loss") is not None

        # pylint: disable=line-too-long,bad-continuation
        assert result.get("hierplane_tree") == {
                'text': 'Please could you parse this sentence ?',
                'root': {
                        'word': 'Please', 'nodeType': 'det', 'attributes': ['UH'], 'link': 'det', 'spans': [{'start': 0, 'end': 7}],
                        'children': [
                                {'word': 'could', 'nodeType': 'nummod', 'attributes': ['MD'], 'link': 'nummod', 'spans': [{'start': 7, 'end': 13}]},
                                {'word': 'you', 'nodeType': 'nummod', 'attributes': ['PRP'], 'link': 'nummod', 'spans': [{'start': 13, 'end': 17}]},
                                {'word': 'parse', 'nodeType': 'nummod', 'attributes': ['VB'], 'link': 'nummod', 'spans': [{'start': 17, 'end': 23}]},
                                {'word': 'this', 'nodeType': 'nummod', 'attributes': ['DT'], 'link': 'nummod', 'spans': [{'start': 23, 'end': 28}]},
                                {'word': 'sentence', 'nodeType': 'nummod', 'attributes': ['NN'], 'link': 'nummod', 'spans': [{'start': 28, 'end': 37}]},
                                {'word': '?', 'nodeType': 'nummod', 'attributes': ['.'], 'link': 'nummod', 'spans': [{'start': 37, 'end': 39}]}]},
                                'nodeTypeToStyle': {'root': ['color5', 'strong'], 'dep': ['color5', 'strong'], 'nsubj': ['color1'], 'nsubjpass': ['color1'],
                                                    'csubj': ['color1'], 'csubjpass': ['color1'], 'pobj': ['color2'], 'dobj': ['color2'],
                                                    'iobj': ['color2'], 'mark': ['color2'], 'pcomp': ['color2'], 'xcomp': ['color2'],
                                                    'ccomp': ['color2'], 'acomp': ['color2'], 'aux': ['color3'], 'cop': ['color3'], 'det': ['color3'],
                                                    'conj': ['color3'], 'cc': ['color3'], 'prep': ['color3'], 'number': ['color3'],
                                                    'possesive': ['color3'], 'poss': ['color3'], 'discourse': ['color3'], 'expletive': ['color3'],
                                                    'prt': ['color3'], 'advcl': ['color3'], 'mod': ['color4'], 'amod': ['color4'], 'tmod': ['color4'],
                                                    'quantmod': ['color4'], 'npadvmod': ['color4'], 'infmod': ['color4'], 'advmod': ['color4'], 'appos': ['color4'],
                                                    'nn': ['color4'], 'neg': ['color0'], 'punct': ['color0']},
                                'linkToPosition': {'nsubj': 'left', 'nsubjpass': 'left', 'csubj': 'left', 'csubjpass': 'left',
                                                   'pobj': 'right', 'dobj': 'right', 'iobj': 'right', 'pcomp': 'right', 'xcomp': 'right',
                                                   'ccomp': 'right', 'acomp': 'right'}}
        # pylint: enable=line-too-long,bad-continuation
    def test_batch_prediction(self):
        inputs = [
                {
                        "sentence": "What kind of test succeeded on its first attempt?",
                },
                {
                        "sentence": "What kind of test succeeded on its first attempt at batch processing?",
                }
        ]

        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser'
                               / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

        results = predictor.predict_batch_json(inputs)
        assert len(results) == 2

        for result in results:
            sequence_length = sum(result.get("mask")) - 1
            heads = result.get("heads")
            assert heads is not None
            assert isinstance(heads, list)
            assert all(isinstance(x, int) for x in heads)
            head_tags = result.get("head_tags")
            assert head_tags is not None
            assert isinstance(head_tags, list)
            assert all(isinstance(x, int) for x in head_tags)

            predicted_heads = result.get("predicted_heads")
            assert len(predicted_heads) == sequence_length

            predicted_dependencies = result.get("predicted_dependencies")
            assert len(predicted_dependencies) == sequence_length
            assert isinstance(predicted_dependencies, list)
            assert all(isinstance(x, str) for x in predicted_dependencies)
