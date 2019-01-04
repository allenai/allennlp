# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class TestQuarelParserPredictor(AllenNlpTestCase):

    def test_answer_present(self):
        inputs = {
                'question':  'Mike was snowboarding on the snow and hit a piece of ice. He went much faster on the ice because _____ is smoother. (A) snow (B) ice',  # pylint: disable=line-too-long
                'world_literals': {'world1': 'snow', 'world2': 'ice'},  # Added to avoid world tagger
                'qrspec': '[smoothness, +speed]',
                'entitycues': 'smoothness: smoother\nspeed:faster'
        }

        archive_path = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'zeroshot' / 'serialization' / 'model.tar.gz'  # pylint: disable=line-too-long
        archive = load_archive(archive_path)
        predictor = Predictor.from_archive(archive, 'quarel-parser')

        result = predictor.predict_json(inputs)
        answer_index = result.get('answer_index')
        assert answer_index is not None

        # Check input modality where entity cues are not given
        del inputs['entitycues']
        result = predictor.predict_json(inputs)
        answer_index = result.get('answer_index')
        assert answer_index is not None
