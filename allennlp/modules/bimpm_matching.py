"""
Multi-perspective matching layer
"""

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import FromParams
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


def masked_max(vector: torch.Tensor,
               mask: torch.Tensor,
               dim: int,
               keepdim: bool = False,
               min_val: float = -1e7) -> torch.Tensor:
    """
    To calculate max along certain dimensions on masked values

    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate max, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        the dimension to calculate max
    keepdim : ``bool``
        Whether to keep dimension
    min_val : ``float``
        The minimal value for paddings

    Returns
    -------
    A ``torch.Tensor`` of including the maximum values.
    """
    replaced_vector = vector + (1.0 - mask) * min_val
    max_value, _ = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value


def masked_mean(vector: torch.Tensor,
                mask: torch.Tensor,
                dim: int,
                keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    Parameters
    ----------
    vector : ``torch.Tensor``
        The vector to calculate mean, assume unmasked parts are already zeros
    mask : ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    dim : ``int``
        the dimension to calculate mean
    keepdim : ``bool``
        Whether to keep dimension
    eps : ``float``
        A small value to avoid zero division problem.

    Returns
    -------
    A ``torch.Tensor`` of including the mean values.
    """
    value_sum = torch.sum(vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=eps)


def mpm(vector1: torch.Tensor,
        vector2: torch.Tensor,
        weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate multi-perspective cosine matching between time-steps of vectors
    of the same length.

    Parameters
    ----------
    vector1 : ``torch.Tensor``
        A tensor of shape ``(batch, seq_len, hidden_size)``
    vector2 : ``torch.Tensor``
        A tensor of shape ``(batch, seq_len or 1, hidden_size)``
    weight : ``torch.Tensor``
        A tensor of shape ``(num_perspective, hidden_size)``

    Returns
    -------
    A tuple of two tensors consisting multi-perspective matching results.
    The first one is of the shape (batch, seq_len, 1), the second one is of shape
    (batch, seq_len, num_perspective)
    """
    assert vector1.size(0) == vector2.size(0)
    assert weight.size(1) == vector1.size(2) == vector1.size(2)

    # (batch, seq_len, 1)
    mv_one = F.cosine_similarity(vector1, vector2, 2).unsqueeze(2)

    # (1, 1, num_perspective, hidden_size)
    weight = weight.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspective, hidden_size)
    vector1 = weight * vector1.unsqueeze(2)
    vector2 = weight * vector2.unsqueeze(2)

    mv_many = F.cosine_similarity(vector1, vector2, dim=3)

    return mv_one, mv_many


def mpm_pairwise(vector1: torch.Tensor,
                 vector2: torch.Tensor,
                 weight: torch.Tensor,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate multi-perspective cosine matching between each time step of
    one vector and each time step of another vector.

    Parameters
    ----------
    vector1 : ``torch.Tensor``
        A tensor of shape ``(batch, seq_len1, hidden_size)``
    vector2 : ``torch.Tensor``
        A tensor of shape ``(batch, seq_len2, hidden_size)``
    weight : ``torch.Tensor``
        A tensor of shape ``(num_perspective, hidden_size)``
    eps : ``float`` optional, (default = 1e-8)
        A small value to avoid zero division problem

    Returns
    -------
    A tensor of shape (batch, seq_len1, seq_len2, num_perspective) consisting
    multi-perspective matching resultsof the shape
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
    return (mul_result / norm_value.clamp(min=eps)).permute(0, 2, 3, 1)


class BiMPMMatching(nn.Module, FromParams):
    """
    This ``Module`` implements the matching layer of BiMPM model described in `Bilateral
    Multi-Perspective Matching for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_
    by Zhiguo Wang et al., 2017.
    Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
    `PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

    Parameters
    ----------
    is_forward : ``bool``, required
        Whether the matching is for forward sequence or backward sequence
    hidden_dim : ``int``, optional (default = 100)
        The hidden dimension of the representations
    num_perspective : ``int``, optional (default = 20)
        The number of perspective for matching
    share_weight_between_directions : ``bool``, optional (default = True)
        If True, share weight between premise to hypothesis and hypothesisto premise,
        useful for non-symmetric tasks
    wo_full_match : ``bool``, optional (default = False)
        If True, exclude full match
    wo_maxpool_match : ``bool``, optional (default = False)
        If True, exclude max_pool match
    wo_attentive_match : ``bool``, optional (default = False)
        If True, exclude attentive match
    wo_max_attentive_match : ``bool``, optional (default = False)
        If True, exclude max attentive match
    """
    def __init__(self,
                 is_forward: bool,
                 hidden_dim: int = 100,
                 num_perspective: int = 20,
                 share_weight_between_directions: bool = True,
                 wo_full_match: bool = False,
                 wo_maxpool_match: bool = False,
                 wo_attentive_match: bool = False,
                 wo_max_attentive_match: bool = False) -> None:
        super(BiMPMMatching, self).__init__()

        self.is_forward = is_forward
        self.hidden_dim = hidden_dim
        self.num_perspective = num_perspective
        self.mv_idx_increment = int(not share_weight_between_directions)

        self.wo_full_match = wo_full_match
        self.wo_maxpool_match = wo_maxpool_match
        self.wo_attentive_match = wo_attentive_match
        self.wo_max_attentive_match = wo_max_attentive_match

        if share_weight_between_directions:
            num_matching = int(not wo_full_match) + \
                           int(not wo_maxpool_match) + \
                           int(not wo_attentive_match) + \
                           int(not wo_max_attentive_match)
        else:
            num_matching = int(not wo_full_match) * 2 + \
                           int(not wo_maxpool_match) + \
                           int(not wo_attentive_match) * 2 + \
                           int(not wo_max_attentive_match) * 2

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
                mask_h: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # pylint: disable=arguments-differ
        """
        Given the forward (or backward) representations of premise and hypothesis, apply four bilateral
        matching functions between premise and hypothesis in one direction.

        Parameters
        ----------
        context_p : ``torch.Tensor``
            Tensor of shape (batch_size, seq_len1, hidden_dim) representing the encoding of premise.
        mask_p : ``torch.Tensor``
            Binary Tensor of shape (batch_size, seq_len1), indicating which
            positions in premise are padding (0) and which are not (1).
        context_h : ``torch.Tensor``
            Tensor of shape (batch_size, seq_len2, hidden_dim) representing the encoding of hypothesis.
        mask_h : ``torch.Tensor``
            Binary Tensor of shape (batch_size, seq_len2), indicating which
            positions in hypothesis are padding (0) and which are not (1).

        Returns
        -------
        A tuple of matching vectors for premise and hypothesis (mv_p, mv_h). Each of which is a list of
        matching vectors of shape (batch, seq_len, num_perspective or 1)

        """
        assert (not mask_h.requires_grad) and (not mask_p.requires_grad)
        assert context_p.size(-1) == context_h.size(-1) == self.hidden_dim

        # (batch,)
        len_p = get_lengths_from_binary_sequence_mask(mask_p)
        len_h = get_lengths_from_binary_sequence_mask(mask_h)

        # (batch, seq_len*)
        mask_p, mask_h = mask_p.float(), mask_h.float()

        # explicitly set masked weights to zero
        # (batch_size, seq_len*, hidden_size)
        context_p = context_p * mask_p.unsqueeze(-1)
        context_h = context_h * mask_h.unsqueeze(-1)

        # array to keep the matching vectors for premise and hypothesis
        mv_p, mv_h = [], []

        # Step 0. unweighted cosine
        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = F.cosine_similarity(context_p.unsqueeze(-2), context_h.unsqueeze(-3), dim=3)

        # (batch, seq_len*, 1)
        cosine_max_p = masked_max(cosine_sim, mask_h.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_p = masked_mean(cosine_sim, mask_h.unsqueeze(-2), dim=2, keepdim=True)
        cosine_max_h = masked_max(cosine_sim.permute(0, 2, 1), mask_p.unsqueeze(-2), dim=2, keepdim=True)
        cosine_mean_h = masked_mean(cosine_sim.permute(0, 2, 1), mask_p.unsqueeze(-2), dim=2, keepdim=True)

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
                # (batch, 1, hidden_dim)
                last_token_p = (len_p - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
                last_token_h = (len_h - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)

                context_p_last = context_p.gather(1, last_token_p)
                context_h_last = context_h.gather(1, last_token_h)
            else:
                context_p_last = context_p[:, 0:1, :]
                context_h_last = context_h[:, 0:1, :]

            # (batch, seq_len*, num_perspective)
            mv_p_full = mpm(context_p, context_h_last, self.params[mv_idx])
            mv_h_full = mpm(context_h, context_p_last, self.params[mv_idx + self.mv_idx_increment])

            mv_p.extend(mv_p_full)
            mv_h.extend(mv_h_full)

            mv_idx += 1 + self.mv_idx_increment

        # Step 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if not self.wo_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspective)
            mv_max = mpm_pairwise(context_p, context_h, self.params[mv_idx])

            # (batch, seq_len*, num_perspective)
            mv_p_max = masked_max(mv_max, mask_h.unsqueeze(-2).unsqueeze(-1), dim=2)
            mv_p_mean = masked_mean(mv_max, mask_h.unsqueeze(-2).unsqueeze(-1), dim=2)
            mv_h_max = masked_max(mv_max.permute(0, 2, 1, 3), mask_p.unsqueeze(-2).unsqueeze(-1), dim=2)
            mv_h_mean = masked_mean(mv_max.permute(0, 2, 1, 3), mask_p.unsqueeze(-2).unsqueeze(-1), dim=2)

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

        # (batch, seq_len1, seq_len2, hidden_size)
        att_h = context_h.unsqueeze(-3) * cosine_sim.unsqueeze(-1)

        # (batch, seq_len1, seq_len2, hidden_size)
        att_p = context_p.unsqueeze(-2) * cosine_sim.unsqueeze(-1)

        if not self.wo_attentive_match:
            # (batch, seq_len*, hidden_size)
            att_mean_h = F.softmax(att_h.sum(dim=2), 2)
            att_mean_p = F.softmax(att_p.sum(dim=1), 2)

            # (batch, seq_len*, num_perspective)
            mv_p_att_mean = mpm(context_p, att_mean_h, self.params[mv_idx])
            mv_h_att_mean = mpm(context_h, att_mean_p, self.params[mv_idx + self.mv_idx_increment])
            mv_p.extend(mv_p_att_mean)
            mv_h.extend(mv_h_att_mean)

            mv_idx += 1 + self.mv_idx_increment

        # Step 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if not self.wo_max_attentive_match:
            # (batch, seq_len*, hidden_size)
            att_max_h = masked_max(att_h, mask_h.unsqueeze(-2).unsqueeze(-1), dim=2)
            att_max_p = masked_max(att_p.permute(0, 2, 1, 3), mask_p.unsqueeze(-2).unsqueeze(-1), dim=2)

            # (batch, seq_len*, num_perspective)
            mv_p_att_max = mpm(context_p, att_max_h, self.params[mv_idx])
            mv_h_att_max = mpm(context_h, att_max_p, self.params[mv_idx + self.mv_idx_increment])

            mv_p.extend(mv_p_att_max)
            mv_h.extend(mv_h_att_max)

            mv_idx += 1 + self.mv_idx_increment

        return mv_p, mv_h
