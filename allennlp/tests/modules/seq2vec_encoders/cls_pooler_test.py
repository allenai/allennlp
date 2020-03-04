import numpy
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler


class TestClsPooler(AllenNlpTestCase):
    def test_encoder(self):
        embedding = torch.rand(5, 50, 7)
        encoder = ClsPooler(embedding_dim=7)
        pooled = encoder(embedding, mask=None)

        assert list(pooled.size()) == [5, 7]
        numpy.testing.assert_array_almost_equal(embedding[:, 0], pooled)

    def test_cls_at_end(self):
        embedding = torch.arange(20).reshape(5, 4).unsqueeze(-1).expand(5, 4, 7)
        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, True, True],
                [True, False, False, False],
                [True, True, False, False],
            ]
        )
        expected = torch.LongTensor([3, 6, 11, 12, 17]).unsqueeze(-1).expand(5, 7)

        encoder = ClsPooler(embedding_dim=7, cls_is_last_token=True)
        pooled = encoder(embedding, mask=mask)

        assert list(pooled.size()) == [5, 7]
        numpy.testing.assert_array_almost_equal(expected, pooled)
