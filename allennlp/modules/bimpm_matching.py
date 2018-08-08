"""
Multi-perspective matching layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import FromParams
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, replace_masked_values


def div_safe(vector1: torch.Tensor, vector2: torch.Tensor, eps: float = 1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    return vector1 / torch.clamp(vector2, min=eps)


def masked_max(vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False, min_val: float = -1e7):
    """
    :param vector: vector to calculate max
    :param mask: mask of the vector, vector and mask must be broadcastable
    :param dim: the dimension to calculate max
    :param keepdim: whether to keep dimension
    :param min_val: minimal value for paddings
    """
    replaced_vector = replace_masked_values(vector, mask, min_val)
    value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    mask_of_max, _ = mask.expand(vector.shape).max(dim=dim, keepdim=keepdim)
    value.mul_(mask_of_max)
    return value


def masked_mean(vector: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False):
    """
    :param vector: vector to calculate max
    :param mask: mask of the vector, vector and mask must be broadcastable
    :param dim: the dimension to calculate mean
    :param keepdim: whether to keep dimension
    """
    value_sum = torch.sum(vector * mask, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.expand(vector.shape), dim=dim, keepdim=keepdim)
    return div_safe(value_sum, value_count.float())


def mpm(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate multi-perspective cosine matching between time-steps of vectors
    of the same length.

    :param vector1: (batch, seq_len, hidden_size)
    :param vector2: (batch, seq_len, hidden_size)
    :param weight: (num_perspective, hidden_size)
    :return: (batch, seq_len, 1), (batch, seq_len, num_perspective)
    """
    num_perspective = weight.size(0)

    assert vector1.shape == vector2.shape and weight.size(1) == vector1.size(2)

    # (batch, seq_len, 1)
    mv_1 = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)

    # (1, 1, num_perspective, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspective, hidden_size)
    vector1 = weight * vector1.unsqueeze(2).expand(-1, -1, num_perspective, -1)
    vector2 = weight * vector2.unsqueeze(2).expand(-1, -1, num_perspective, -1)

    mv_many = F.cosine_similarity(vector1, vector2, dim=3)

    return mv_1, mv_many


def mpm_pairwise(vector1: torch.Tensor, vector2: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Calculate multi-perspective cosine matching between each time step of
    one vector and each time step of another vector.

    :param vector1: (batch, seq_len1, hidden_size)
    :param vector2: (batch, seq_len2, hidden_size)
    :param weight: (num_perspective, hidden_size)
    :return: (batch, seq_len1, seq_len2, num_perspective)
    """

    num_perspective = weight.size(0)

    # (1, num_perspective, 1, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(2)

    # (batch, num_perspective, seq_len*, hidden_size)
    vector1 = weight * vector1.unsqueeze(1).expand(-1, num_perspective, -1, -1)
    vector2 = weight * vector2.unsqueeze(1).expand(-1, num_perspective, -1, -1)

    # (batch, num_perspective, seq_len*, 1)
    v1_norm = vector1.norm(p=2, dim=3, keepdim=True)
    v2_norm = vector2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspective, seq_len1, seq_len2)
    mul_result = torch.matmul(vector1, vector2.transpose(2, 3))
    norm_value = v1_norm * v2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspective)
    return div_safe(mul_result, norm_value).permute(0, 2, 3, 1)


def cosine_pairwise(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity between each time step of
    one vector and each time step of another vector.

    :param vector1: (batch, seq_len1, hidden_size)
    :param vector2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """

    batch_size = vector1.size(0)
    assert batch_size == vector2.size(0) and vector1.size(2) == vector2.size(2)

    # (batch, seq_len1, 1)
    vector1_norm = vector1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    vector2_norm = vector2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

    # (batch, seq_len1, seq_len2)
    mul_result = torch.bmm(vector1, vector2.permute(0, 2, 1))
    norm_value = vector1_norm * vector2_norm

    result = div_safe(mul_result, norm_value)

    return result


class BiMPMMatching(nn.Module, FromParams):
    def __init__(self,
                 is_forward: bool,
                 hidden_dim: int = 100,
                 num_perspective: int = 20,
                 wo_full_match: bool = False,
                 wo_maxpool_match: bool = False,
                 wo_attentive_match: bool = False,
                 wo_max_attentive_match: bool = False) -> None:
        super(BiMPMMatching, self).__init__()

        self.is_forward = is_forward
        self.hidden_dim = hidden_dim
        self.num_perspective = num_perspective

        self.wo_full_match = wo_full_match
        self.wo_maxpool_match = wo_maxpool_match
        self.wo_attentive_match = wo_attentive_match
        self.wo_max_attentive_match = wo_max_attentive_match

        num_matching = 4 - sum([wo_full_match, wo_maxpool_match, wo_attentive_match, wo_max_attentive_match])
        if num_matching <= 0:
            raise ConfigurationError("At least one of the matching method should be enabled")

        params = [nn.Parameter(torch.rand(num_perspective, hidden_dim)) for i in range(num_matching)]
        self.params = nn.ParameterList(params)

        # calculate the output dimension for each of the matching vector
        dim = 2  # cosine max and cosine min
        dim += 0 if wo_full_match else num_perspective + 1  # full match
        dim += 0 if wo_maxpool_match else num_perspective * 2  # max pool match and mean pool match
        dim += 0 if wo_attentive_match else num_perspective + 1  # attentive match
        dim += 0 if wo_max_attentive_match else num_perspective + 1  # max attentive match
        self.output_dim = dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                context_p: torch.Tensor,
                mask_p: torch.Tensor,
                context_h: torch.Tensor,
                mask_h: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """
        Given the representations of two sentences from BiLSTM, apply four bilateral
        matching functions between premise and hypothesis in one direction

        Parameters
        ----------
        :param context_p: Tensor of shape (batch_size, seq_len, context_rnn_hidden_dim)
            representing premise as encoded by the forward and backward layer of a BiLSTM.
        :param mask_p: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in premise are padding (0) and which are not (1).
        :param context_h: Tensor of shape (batch_size, seq_len, context_rnn_hidden_dim)
            representing hypothesis as encoded by the forward and backward layer of a BiLSTM.
        :param mask_h: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in hypothesis are padding (0) and which are not (1).
        :return (mv_p, mv_h): Matching vectors for premise and hypothesis, each of shape
            (batch, seq_len, num_perspective * num_matching)
        """

        # (batch, seq_len, hidden_dim)
        assert context_p.size(-1) == context_h.size(-1) == self.hidden_dim
        seq_len_p, seq_len_h = context_p.size(1), context_h.size(1)

        mask_p, mask_h = mask_p.float(), mask_h.float()

        # mask for pairwise matching
        # (batch, seq_len1, seq_len2)
        mask_p_h = torch.bmm(mask_p.unsqueeze(-1), mask_h.unsqueeze(-2))
        mask_h_p = mask_p_h.permute(0, 2, 1)

        # explicitly set masked weights to zero
        context_p.mul_(mask_p.unsqueeze(-1))
        context_h.mul_(mask_h.unsqueeze(-1))

        # array to keep the matching vectors for premise and hypothesis
        mv_p, mv_h = [], []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = cosine_pairwise(context_p, context_h)

        # (batch, seq_len*, 1)
        cosine_max_p = masked_max(cosine_sim, mask_p_h, dim=2, keepdim=True)
        cosine_mean_p = masked_mean(cosine_sim, mask_p_h, dim=2, keepdim=True)
        cosine_max_h = masked_max(cosine_sim.permute(0, 2, 1), mask_h_p, dim=2, keepdim=True)
        cosine_mean_h = masked_mean(cosine_sim.permute(0, 2, 1), mask_h_p, dim=2, keepdim=True)

        mv_p.extend([cosine_max_p, cosine_mean_p])
        mv_h.extend([cosine_max_h, cosine_mean_h])

        mv_idx = 0

        # Step 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if not self.wo_full_match:

            # (batch, 1, hidden_size)
            if self.is_forward:
                # (batch,)
                len_p = get_lengths_from_binary_sequence_mask(mask_p)
                len_h = get_lengths_from_binary_sequence_mask(mask_h)

                # (batch, 1, hidden_dim)
                last_token_p = torch.clamp(len_p - 1, min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
                last_token_h = torch.clamp(len_h - 1, min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)

                context_p_last = context_p.gather(1, last_token_p)
                context_h_last = context_h.gather(1, last_token_h)
            else:
                context_p_last = context_p[:, 0:1, :]
                context_h_last = context_h[:, 0:1, :]

            # (batch, seq_len*, num_perspective)
            mv_p_full = mpm(context_p, context_h_last.expand(-1, seq_len_p, -1), self.params[mv_idx])
            mv_h_full = mpm(context_h, context_p_last.expand(-1, seq_len_h, -1), self.params[mv_idx])

            mv_p.extend(mv_p_full)
            mv_h.extend(mv_h_full)

            mv_idx += 1

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if not self.wo_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspective)
            mv_max = mpm_pairwise(context_p, context_h, self.params[mv_idx])

            # (batch, seq_len1, num_perspective)
            mv_p_max = masked_max(mv_max, mask_p_h.unsqueeze(-1), dim=2)
            mv_p_mean = masked_mean(mv_max, mask_p_h.unsqueeze(-1), dim=2)

            # (batch, seq_len2, seq_len1, num_perspective)
            mv_max = mv_max.permute(0, 2, 1, 3)

            # (batch, seq_len2, num_perspective)
            mv_h_max = masked_max(mv_max, mask_h_p.unsqueeze(-1), dim=2)
            mv_h_mean = masked_mean(mv_max, mask_h_p.unsqueeze(-1), dim=2)

            mv_p.extend([mv_p_max, mv_p_mean])
            mv_h.extend([mv_h_max, mv_h_mean])

            mv_idx += 1

        # Step 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h = context_h.unsqueeze(1) * cosine_sim.unsqueeze(3)

        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p = context_p.unsqueeze(2) * cosine_sim.unsqueeze(3)

        if not self.wo_attentive_match:
            # (batch, seq_len*, hidden_size) / (batch, seq_len*, 1) ->
            # (batch, seq_len*, hidden_size)
            att_mean_h = div_safe(att_h.sum(dim=2), cosine_sim.sum(dim=2, keepdim=True))
            att_mean_p = div_safe(att_p.sum(dim=1), cosine_sim.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len*, num_perspective)
            mv_p_att_mean = mpm(context_p, att_mean_h, self.params[mv_idx])
            mv_h_att_mean = mpm(context_h, att_mean_p, self.params[mv_idx])
            mv_p.extend(mv_p_att_mean)
            mv_h.extend(mv_h_att_mean)

            mv_idx += 1

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if not self.wo_max_attentive_match:
            # (batch, seq_len*, hidden_size)
            att_max_h = masked_max(att_h, mask_p_h.unsqueeze(-1), dim=2)
            att_max_p = masked_max(att_p.permute(0, 2, 1, 3), mask_h_p.unsqueeze(-1), dim=2)

            # (batch, seq_len*, num_perspective)
            mv_p_att_max = mpm(context_p, att_max_h, self.params[mv_idx])
            mv_h_att_max = mpm(context_h, att_max_p, self.params[mv_idx])

            mv_p.extend(mv_p_att_max)
            mv_h.extend(mv_h_att_max)

            mv_idx += 1

        return mv_p, mv_h
