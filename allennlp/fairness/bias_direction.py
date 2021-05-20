"""
A suite of differentiable methods to compute the bias direction
or concept subspace representing binary protected variables.
"""

import torch
import sklearn
import numpy as np

from allennlp.common.checks import ConfigurationError


class BiasDirection:
    """
    Parent class for bias direction classes.

    # Parameters

    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation.
    """

    def __init__(self, requires_grad: bool = False):
        self.requires_grad = requires_grad

    def _normalize_bias_direction(self, bias_direction: torch.Tensor):
        return bias_direction / torch.linalg.norm(bias_direction)


class PCABiasDirection(BiasDirection):
    """
    PCA-based bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) using PCA. This subspace minimizes the sum of
    squared distances from all seed word embeddings.

    !!! Note
        It is uncommon to utilize more than one direction to represent a concept.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(self, seed_embeddings: torch.Tensor):
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        seed_embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) containing seed word embeddings related to
            a concept. For example, if the concept is gender, seed_embeddings could contain embeddings
            for words like "man", "king", "brother", "woman", "queen", "sister", etc.

        # Returns

        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.
        """

        # Some sanity checks
        if seed_embeddings.ndim < 2:
            raise ConfigurationError("seed_embeddings1 must have at least two dimensions.")

        with torch.set_grad_enabled(self.requires_grad):
            # pca_lowrank centers the embeddings by default
            # There will be two dimensions when applying PCA to
            # definitionally-gendered words: 1) the gender direction,
            # 2) all other directions, with the gender direction being principal.
            _, _, V = torch.pca_lowrank(seed_embeddings, q=2)
            # get top principal component
            bias_direction = V[:, 0]
            return self._normalize_bias_direction(bias_direction)


class PairedPCABiasDirection(BiasDirection):
    """
    Paired-PCA-based bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) as the first principle component of the
    difference vectors between seed word embedding pairs.

    !!! Note
        It is uncommon to utilize more than one direction to represent a concept.

    Based on: T. Bolukbasi, K. W. Chang, J. Zou, V. Saligrama, and A. Kalai. [Man is to
    computer programmer as woman is to homemaker? debiasing word embeddings]
    (https://api.semanticscholar.org/CorpusID:1704893).
    In ACM Transactions of Information Systems, 2016.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(self, seed_embeddings1: torch.Tensor, seed_embeddings2: torch.Tensor):
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        seed_embeddings1 : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) containing seed word
            embeddings related to a concept group. For example, if the concept is gender,
            seed_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc.

        seed_embeddings2: `torch.Tensor`
            A tensor of the same size as seed_embeddings1 containing seed word
            embeddings related to a different group for the same concept. For example,
            seed_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc.

        !!! Note
            For Paired-PCA, the embeddings at the same positions in each of seed_embeddings1 and
            seed_embeddings2 are expected to form seed word pairs. For example, if the concept
            is gender, the embeddings for ("man", "woman"), ("king", "queen"), ("brother", "sister"), etc.
            should be at the same positions in seed_embeddings1 and seed_embeddings2.

        !!! Note
            All tensors are expected to be on the same device.

        # Returns

        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.
        """

        # Some sanity checks
        if seed_embeddings1.size() != seed_embeddings2.size():
            raise ConfigurationError("seed_embeddings1 and seed_embeddings2 must be the same size.")
        if seed_embeddings1.ndim < 2:
            raise ConfigurationError(
                "seed_embeddings1 and seed_embeddings2 must have at least two dimensions."
            )

        with torch.set_grad_enabled(self.requires_grad):
            paired_embeddings = seed_embeddings1 - seed_embeddings2
            _, _, V = torch.pca_lowrank(
                paired_embeddings,
                q=min(paired_embeddings.size(0), paired_embeddings.size(1)) - 1,
            )
            bias_direction = V[:, 0]
            return self._normalize_bias_direction(bias_direction)


class TwoMeansBiasDirection(BiasDirection):
    """
    Two-means bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) as the normalized difference vector of the
    averages of seed word embedding sets.

    !!! Note
        It is uncommon to utilize more than one direction to represent a concept.

    Based on: Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
    (https://api.semanticscholar.org/CorpusID:59158788). AISTATS.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(self, seed_embeddings1: torch.Tensor, seed_embeddings2: torch.Tensor):
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        seed_embeddings1 : `torch.Tensor`
            A tensor of size (embeddings1_batch_size, ..., dim) containing seed word
            embeddings related to a specific concept group. For example, if the concept is gender,
            seed_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc.
        seed_embeddings2: `torch.Tensor`
            A tensor of size (embeddings2_batch_size, ..., dim) containing seed word
            embeddings related to a different group for the same concept. For example,
            seed_embeddings2 could contain embeddings for linguistically feminine words, , e.g.
            "woman", "queen", "sister", etc.

        !!! Note
            seed_embeddings1 and seed_embeddings2 need NOT be the same size. Furthermore,
            the embeddings at the same positions in each of seed_embeddings1 and seed_embeddings2
            are NOT expected to form seed word pairs.

        !!! Note
            All tensors are expected to be on the same device.

        # Returns

        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.
        """
        # Some sanity checks
        if seed_embeddings1.ndim < 2 or seed_embeddings2.ndim < 2:
            raise ConfigurationError(
                "seed_embeddings1 and seed_embeddings2 must have at least two dimensions."
            )
        if seed_embeddings1.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError("All seed embeddings must have same dimensionality.")

        with torch.set_grad_enabled(self.requires_grad):
            seed_embeddings1_mean = torch.mean(seed_embeddings1, dim=0)
            seed_embeddings2_mean = torch.mean(seed_embeddings2, dim=0)
            bias_direction = seed_embeddings1_mean - seed_embeddings2_mean
            return self._normalize_bias_direction(bias_direction)


class ClassificationNormalBiasDirection(BiasDirection):
    """
    Classification normal bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) as the direction perpendicular to the classification
    boundary of a linear support vector machine fit to classify seed word embedding sets.

    !!! Note
        It is uncommon to utilize more than one direction to represent a concept.

    Based on: Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020).
    [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection]
    (https://api.semanticscholar.org/CorpusID:215786522). ArXiv, abs/2004.07667.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, seed_embeddings1: torch.Tensor, seed_embeddings2: torch.Tensor):
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        seed_embeddings1 : `torch.Tensor`
            A tensor of size (embeddings1_batch_size, ..., dim) containing seed word
            embeddings related to a specific concept group. For example, if the concept is gender,
            seed_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc.
        seed_embeddings2: `torch.Tensor`
            A tensor of size (embeddings2_batch_size, ..., dim) containing seed word
            embeddings related to a different group for the same concept. For example,
            seed_embeddings2 could contain embeddings for linguistically feminine words, , e.g.
            "woman", "queen", "sister", etc.

        !!! Note
            seed_embeddings1 and seed_embeddings2 need NOT be the same size. Furthermore,
            the embeddings at the same positions in each of seed_embeddings1 and seed_embeddings2
            are NOT expected to form seed word pairs.

        !!! Note
            All tensors are expected to be on the same device.

        !!! Note
            This bias direction method is NOT differentiable.

        # Returns

        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.
        """

        # Some sanity checks
        if seed_embeddings1.ndim < 2 or seed_embeddings2.ndim < 2:
            raise ConfigurationError(
                "seed_embeddings1 and seed_embeddings2 must have at least two dimensions."
            )
        if seed_embeddings1.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError("All seed embeddings must have same dimensionality.")

        device = seed_embeddings1.device
        seed_embeddings1 = seed_embeddings1.flatten(end_dim=-2).detach().cpu().numpy()
        seed_embeddings2 = seed_embeddings2.flatten(end_dim=-2).detach().cpu().numpy()

        X = np.vstack([seed_embeddings1, seed_embeddings2])
        Y = np.concatenate([[0] * seed_embeddings1.shape[0], [1] * seed_embeddings2.shape[0]])

        classifier = sklearn.svm.SVC(kernel="linear").fit(X, Y)
        bias_direction = torch.Tensor(classifier.coef_[0]).to(device)

        return self._normalize_bias_direction(bias_direction)
