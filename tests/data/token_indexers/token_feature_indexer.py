# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import FeatureIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
import numpy as np

def lowercase_feature_extractor(token: Token):
    if token.text[0].isupper():
        return np.array([1.0], dtype=np.float32)
    else:
        return np.array([0.0], dtype=np.float32)

class TestPosTagIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestPosTagIndexer, self).setUp()
        self.tokenizer = SpacyWordSplitter(pos_tags=True)



    def test_count_extract_features(self):
        tokens = self.tokenizer.split_words("This is a sentence.")
        tokens = [Token("<S>")] + [t for t in tokens] + [Token("</S>")]
        indexer = FeatureIndexer(feature_extractor=lowercase_feature_extractor,
                                 default_value=np.array([0.0], dtype=np.float32))
        assert indexer.token_to_indices(tokens[1]) == np.array([1.0], dtype=np.float32)
        assert indexer.token_to_indices(tokens[2]) == np.array([0.0], dtype=np.float32)



