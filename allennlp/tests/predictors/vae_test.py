# pylint: disable=no-self-use,invalid-name,protected-access,unused-import

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from allennlp.models.encoder_decoders.vae import VAE
from allennlp.modules.seq2vec_encoders.masked_encoder import MaskedEncoder
from allennlp.predictors.vae import VAEPredictor

from allennlp.common.testing import AllenNlpTestCase

class TestVAEPredictor(AllenNlpTestCase):
    def test_uses_named_inputs_with_vae(self):
        inputs = {"num_to_generate": 2}

        archive = load_archive(self.FIXTURES_ROOT / 'encoder_decoder' / 'vae' /
                               'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'vae')

        results = predictor.predict_json(inputs)

        predicted_tokens = results['predicted_tokens']

        assert predicted_tokens is not None
        assert isinstance(predicted_tokens, list)
        assert all(x for x in predicted_tokens)
        assert all(isinstance(x, str) for sent in predicted_tokens for x in sent)
