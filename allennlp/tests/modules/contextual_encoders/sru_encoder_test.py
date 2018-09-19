# pylint: disable=invalid-name
import numpy as np
import pytest
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.contextual_encoders.sru_encoder import SruEncoder, reverse_padded_sequence


def reverse_sequence_numpy(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    y = x.copy()
    for k in range(x.shape[0]):
        length = int(lengths[k])
        y[k, :length, :] = x[k, :length, :][::-1, :]
    return y


class TestReversePaddedSequence(AllenNlpTestCase):
    def test_reverse_padded_sequence(self):
        x = torch.Tensor(torch.rand(3, 4, 5))
        lengths = [4, 2, 3]
        y = reverse_padded_sequence(x, lengths, batch_first=True)
        expected_y = reverse_sequence_numpy(x.data.numpy(), lengths)

        assert np.allclose(y.data.numpy(), expected_y)

class TestSruEncoder(AllenNlpTestCase):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device registered.")
    def test_sru_encoder(self):
        se = SruEncoder(dim=32,
                        num_layers=1,
                        dropout=0.1,
                        rnn_dropout=0.1,
                        input_dropout=0.2,
                        use_tanh=True)

        # Check that our implementation of bidirectional with masking
        # works properly.
        se.cuda()
        batch_size, timesteps, dim = 5, 10, 32
        token_embeddings = torch.rand(batch_size, timesteps, dim).cuda()
        mask = torch.ones(batch_size, timesteps).cuda()
        mask[0, 7:] = 0
        mask[1, 5:] = 0
        mask[2, -1] = 0
        mask[3, 2:] = 0

        se.eval()
        output = se(token_embeddings, mask)

        # Now initialize a single bidirectional SRU with these weights
        # and compare.
        from sru.cuda_functional import SRU
        rnn = SRU(32, 32,
                  num_layers=1,
                  dropout=0.1,
                  rnn_dropout=0.1,
                  use_tanh=1,
                  use_relu=0,
                  bidirectional=False)
        rnn.cuda()
        rnn.eval()
        rnn_cell = rnn.rnn_lst[0]
        rnn_cell.weight = se.forward_0.weight
        rnn_cell.bias = se.forward_0.bias
        sru_forward_output_t, _ = rnn_cell(token_embeddings.transpose(0, 1))
        sru_forward_output = sru_forward_output_t.transpose(0, 1)

        self.assertTrue(np.allclose(
            output[:, :, :dim].data.cpu().numpy(),
            sru_forward_output.data.cpu().numpy())
        )

        # Now the backward direction
        rnn_cell.weight = se.backward_0.weight
        rnn_cell.bias = se.backward_0.bias

        lengths = mask.sum(dim=1).cpu().data.numpy().astype('int32')

        numpy_token_embeddings = token_embeddings.data.cpu().numpy().copy()
        token_embeddings_back = reverse_sequence_numpy(
            numpy_token_embeddings, lengths)

        token_embeddings_back_var = torch.from_numpy(token_embeddings_back).transpose(0, 1).cuda()
        sru_backward_output_t, _ = rnn_cell(token_embeddings_back_var)
        sru_backward_output_reverse = sru_backward_output_t.\
            transpose(0, 1).data.cpu().numpy()
        sru_backward_output = reverse_sequence_numpy(
            sru_backward_output_reverse, lengths)

        assert np.allclose(output[:, :, dim:].data.cpu().numpy(),
                           sru_backward_output)

