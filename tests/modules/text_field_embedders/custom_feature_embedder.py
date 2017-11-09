from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, Dataset, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, FeatureIndexer
import numpy as np
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.bypass_embedder import BypassEmbedder
from allennlp.nn.util import arrays_to_variables


def lowercase_feature_extractor(token: Token):
    if token.text[0].isupper():
        return np.array([1.0], dtype=np.float32)
    else:
        return np.array([0.0], dtype=np.float32)



class TestCustomFeatureFieldEmbedder(AllenNlpTestCase):
    def setUp(self):
        super(TestCustomFeatureFieldEmbedder, self).setUp()
        self.indexers = {
            'tokens': SingleIdTokenIndexer(),
            "is_capitalized": FeatureIndexer(feature_extractor=lowercase_feature_extractor,
                                         default_value=np.array([0.0], dtype=np.float32)),
        }


    def test_correct_dimensionality(self):
        sentence1 = TextField([Token(t) for t in ["One", "two"]],
                          token_indexers=self.indexers)
        instance1 = Instance({"sentence": sentence1})
        vocab = Vocabulary()
        vocab.add_token_to_namespace("One")
        vocab.add_token_to_namespace("two")

        word_embedding_size = 10
        word_embedding = Embedding(num_embeddings=vocab.get_vocab_size("tokens"),
                                   embedding_dim=word_embedding_size)
        embedder = BasicTextFieldEmbedder({"tokens": word_embedding,
                                                "is_capitalized": BypassEmbedder(dimensionality=1)})
        dataset = Dataset([instance1])
        dataset.index_instances(vocab)

        arrays = dataset.as_array_dict(dataset.get_padding_lengths())
        torch_arrays = arrays_to_variables(arrays)

        embedded_text = embedder(torch_arrays["sentence"])
        assert embedded_text.size(2) == word_embedding_size + 1
