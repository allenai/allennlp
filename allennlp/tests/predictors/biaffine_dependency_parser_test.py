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

        words = result.get("words")
        predicted_heads = result.get("predicted_heads")
        assert len(predicted_heads) == len(words)

        predicted_dependencies = result.get("predicted_dependencies")
        assert len(predicted_dependencies) == len(words)
        assert isinstance(predicted_dependencies, list)
        assert all(isinstance(x, str) for x in predicted_dependencies)

        assert result.get("loss") is not None
        assert result.get("arc_loss") is not None
        assert result.get("tag_loss") is not None

        hierplane_tree = result.get("hierplane_tree")
        hierplane_tree.pop("nodeTypeToStyle")
        hierplane_tree.pop("linkToPosition")
        # pylint: disable=line-too-long,bad-continuation
        assert result.get("hierplane_tree") == {'text': 'Please could you parse this sentence ?',
                                                'root': {'word': 'Please', 'nodeType': 'det', 'attributes': ['INTJ'], 'link': 'det', 'spans': [{'start': 0, 'end': 7}],
                                                    'children': [
                                                            {'word': 'could', 'nodeType': 'nummod', 'attributes': ['VERB'], 'link': 'nummod', 'spans': [{'start': 7, 'end': 13}]},
                                                            {'word': 'you', 'nodeType': 'nummod', 'attributes': ['PRON'], 'link': 'nummod', 'spans': [{'start': 13, 'end': 17}]},
                                                            {'word': 'parse', 'nodeType': 'nummod', 'attributes': ['VERB'], 'link': 'nummod', 'spans': [{'start': 17, 'end': 23}]},
                                                            {'word': 'this', 'nodeType': 'nummod', 'attributes': ['DET'], 'link': 'nummod', 'spans': [{'start': 23, 'end': 28}]},
                                                            {'word': 'sentence', 'nodeType': 'nummod', 'attributes':['NOUN'], 'link': 'nummod', 'spans': [{'start': 28, 'end': 37}]},
                                                            {'word': '?', 'nodeType': 'nummod', 'attributes': ['PUNCT'], 'link': 'nummod', 'spans': [{'start': 37, 'end': 39}]}
                                                            ]
                                                        }
                                               }
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
            sequence_length = len(result.get("words"))
            predicted_heads = result.get("predicted_heads")
            assert len(predicted_heads) == sequence_length

            predicted_dependencies = result.get("predicted_dependencies")
            assert len(predicted_dependencies) == sequence_length
            assert isinstance(predicted_dependencies, list)
            assert all(isinstance(x, str) for x in predicted_dependencies)

    def test_predictor_uses_dataset_reader_to_determine_pos_set(self):
        # pylint: disable=protected-access
        archive = load_archive(self.FIXTURES_ROOT / 'biaffine_dependency_parser'
                               / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'biaffine-dependency-parser')

        inputs = {
                "sentence": "Dogs eat cats.",
        }
        instance_with_ud_pos = predictor._json_to_instance(inputs)
        tags = instance_with_ud_pos.fields["pos_tags"].labels
        assert tags == ['NOUN', 'VERB', 'NOUN', 'PUNCT']

        predictor._dataset_reader.use_language_specific_pos = True

        instance_with_ptb_pos = predictor._json_to_instance(inputs)
        tags = instance_with_ptb_pos.fields["pos_tags"].labels
        assert tags == ['NNS', 'VBP', 'NNS', '.']
