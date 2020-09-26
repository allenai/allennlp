import torch

from allennlp.common import Params
from allennlp.modules import BiMpmMatching
from allennlp.common.testing import AllenNlpTestCase


class TestBiMPMMatching(AllenNlpTestCase):
    def test_forward(self):
        batch = 16
        len1, len2 = 21, 24
        seq_len1 = torch.randint(low=len1 - 10, high=len1 + 1, size=(batch,)).long()
        seq_len2 = torch.randint(low=len2 - 10, high=len2 + 1, size=(batch,)).long()

        mask1 = []
        for w in seq_len1:
            mask1.append([1] * w.item() + [0] * (len1 - w.item()))
        mask1 = torch.tensor(mask1, dtype=torch.bool)
        mask2 = []
        for w in seq_len2:
            mask2.append([1] * w.item() + [0] * (len2 - w.item()))
        mask2 = torch.tensor(mask2, dtype=torch.bool)

        d = 200  # hidden dimension
        n = 20  # number of perspective
        test1 = torch.randn(batch, len1, d)
        test2 = torch.randn(batch, len2, d)
        test1 = test1 * mask1.view(-1, len1, 1).expand(-1, len1, d)
        test2 = test2 * mask2.view(-1, len2, 1).expand(-1, len2, d)

        test1_fw, test1_bw = torch.split(test1, d // 2, dim=-1)
        test2_fw, test2_bw = torch.split(test2, d // 2, dim=-1)

        ml_fw = BiMpmMatching.from_params(Params({"is_forward": True, "num_perspectives": n}))
        ml_bw = BiMpmMatching.from_params(Params({"is_forward": False, "num_perspectives": n}))

        vecs_p_fw, vecs_h_fw = ml_fw(test1_fw, mask1, test2_fw, mask2)
        vecs_p_bw, vecs_h_bw = ml_bw(test1_bw, mask1, test2_bw, mask2)
        vecs_p, vecs_h = (
            torch.cat(vecs_p_fw + vecs_p_bw, dim=2),
            torch.cat(vecs_h_fw + vecs_h_bw, dim=2),
        )

        assert vecs_p.size() == torch.Size([batch, len1, 10 + 10 * n])
        assert vecs_h.size() == torch.Size([batch, len2, 10 + 10 * n])
        assert (
            ml_fw.get_output_dim()
            == ml_bw.get_output_dim()
            == vecs_p.size(2) // 2
            == vecs_h.size(2) // 2
        )
