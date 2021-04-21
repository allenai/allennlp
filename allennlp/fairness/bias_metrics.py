"""

A suite of metrics to quantify how much bias is
encoded by word embeddings and determine the effectiveness
of bias mitigation.

Bias metrics are based on:

1. Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
Science, 356, 183 - 186.

2. Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
(https://api.semanticscholar.org/CorpusID:59158788). AISTATS.

3. Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating
Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701).
ArXiv, abs/1908.09369.

"""

import torch
from allennlp.common.checks import ConfigurationError


class WordEmbeddingAssociationTest:
    """
    Word Embedding Association Test (WEAT) score measures the unlikelihood there is no
    difference between two sets of target words in terms of their relative similarity
    to two sets of attribute words by computing the probability that a random
    permutation of attribute words would produce the observed (or greater) difference
    in sample means. Analog of Implicit Association Test from psychology for word embeddings.

    Based on: Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
    from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
    Science, 356, 183 - 186.
    """

    def __call__(
        self,
        target_embeddings1: torch.Tensor,
        target_embeddings2: torch.Tensor,
        attribute_embeddings1: torch.Tensor,
        attribute_embeddings2: torch.Tensor,
    ) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings1 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings1_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group. For example, if the concept is professions,
            attribute_embeddings1 could contain embeddings for stereotypically male professions, e.g.
            "doctor", "banker", "engineer", etc. Represented as A.

        attribute_embeddings2 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings2_batch_size, ..., dim) containing attribute word
            embeddings related to a different group for the same concept. For example, if the concept is
            professions, attribute_embeddings2 could contain embeddings for stereotypically female
            professions, e.g. "nurse", "receptionist", "homemaker", etc. Represented as B.

        # Returns

        weat_score : `torch.FloatTensor`
            The unlikelihood there is no difference between target_embeddings1 and target_embeddings2 in
            terms of their relative similarity to attribute_embeddings1 and attribute_embeddings2.
            Typical values are around [-1, 1], with values closer to 0 indicating less biased associations.

        """

        # Some sanity checks
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must have at least two dimensions."
            )
        if attribute_embeddings1.ndim < 2 or attribute_embeddings2.ndim < 2:
            raise ConfigurationError(
                "attribute_embeddings1 and attribute_embeddings2 must have at least two dimensions."
            )
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError(
                "target_embeddings1 and target_embeddings2 must be of the same size."
            )
        if attribute_embeddings1.size(dim=-1) != attribute_embeddings2.size(
            dim=-1
        ) or attribute_embeddings1.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError("All embeddings must have the same dimensionality.")

        target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
        target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
        attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
        attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)

        # Normalize
        target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
        target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
        attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
        attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)

        # Compute cosine similarities
        X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
        X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
        Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
        Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
        X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
        X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])

        s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
        s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
        s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
        S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        return s_X_Y_A_B / torch.std(S_X_union_Y_A_B, unbiased=False)


class EmbeddingCoherenceTest:
    pass


class NaturalLanguageInference:
    pass
