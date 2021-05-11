"""
A suite of differentiable methods to mitigate
biases for binary concepts in embeddings.
"""

import torch
import numpy as np
import scipy
import sklearn

from allennlp.common.checks import ConfigurationError


class BiasMitigator:
    """
    Parent class for bias mitigator classes.

    # Parameters

    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation.
    """

    def __init__(self, requires_grad: bool = False):
        self.requires_grad = requires_grad

    def _proj(self, u: torch.Tensor, v: torch.Tensor, normalize: bool = False):
        proj = torch.matmul(u, v.reshape(-1, 1)) * v
        if normalize:
            return proj / torch.dot(v, v)
        return proj

    def _remove_component(
        self, embeddings: torch.Tensor, bias_direction: torch.Tensor, normalize: bool = False
    ):
        return embeddings - self._proj(embeddings, bias_direction, normalize)


class HardBiasMitigator(BiasMitigator):
    """
    Hard bias mitigator. Mitigates bias in embeddings by:

    1. Neutralizing: ensuring protected variable-neutral words remain equidistant
    from the bias direction by removing component of embeddings
    in the bias direction.

    2. Equalizing: ensuring that protected variable-related words are averaged
    out to have the same norm.

    !!! Note
        For a detailed walkthrough and visual descriptions of the steps, please
        refer to Figure 4 in [VERB: Visualizing and Interpreting Bias Mitigation Techniques
        for Word Representations](https://api.semanticscholar.org/CorpusID:233168618).

    Based on: T. Bolukbasi, K. W. Chang, J. Zou, V. Saligrama, and A. Kalai. [Man is to
    computer programmer as woman is to homemaker? debiasing word embeddings]
    (https://api.semanticscholar.org/CorpusID:1704893).
    In ACM Transactions of Information Systems, 2016.

    Description taken from: Goenka, D. (2020). [Tackling Gender Bias in Word Embeddings]
    (https://towardsdatascience.com/tackling-gender-bias-in-word-embeddings-c965f4076a10).

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(
        self,
        evaluation_embeddings: torch.Tensor,
        bias_direction: torch.Tensor,
        equalize_embeddings1: torch.Tensor,
        equalize_embeddings2: torch.Tensor,
    ):
        """

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        # Parameters

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (evaluation_batch_size, ..., dim) of embeddings for which to mitigate bias.
        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace. The words
            that are used to define the bias direction are considered definitionally
            gendered and not modified.
        equalize_embeddings1: `torch.Tensor`
            A tensor of size (equalize_batch_size, ..., dim) containing equalize word
            embeddings related to a group from the concept represented by bias_direction.
            For example, if the concept is gender, equalize_embeddings1 could contain embeddings
            for "boy", "man", "dad", "brother", etc.
        equalize_embeddings2: `torch.Tensor`
            A tensor of size (equalize_batch_size, ..., dim) containing equalize word
            embeddings related to a different group for the same concept. For example,
            equalize_embeddings2 could contain embeddings for "girl", "woman", "mom",
            "sister", etc.

        !!! Note
            The embeddings at the same positions in each of equalize_embeddings1 and
            equalize_embeddings2 are expected to form equalize word pairs. For example, if the concept
            is gender, the embeddings for ("boy", "girl"), ("man", "woman"), ("dad", "mom"),
            ("brother", "sister"), etc. should be at the same positions in equalize_embeddings1
            and equalize_embeddings2.

        !!! Note
            evaluation_embeddings, equalize_embeddings1, and equalize_embeddings2 must have same size
            except for 0th dim (i.e. batch dimension).

        !!! Note
            Please ensure that the words in evaluation_embeddings, equalize_embeddings1, and
            equalize_embeddings2 and those used to compute bias_direction are disjoint.

        !!! Note
            All tensors are expected to be on the same device.

        # Returns

        bias_mitigated_embeddings : `torch.Tensor`
            A tensor of the same size as evaluation_embeddings, equalize_embeddings1, and equalize_embeddings2
            (in this order) stacked.
        """

        # Some sanity checks
        if equalize_embeddings1.size() != equalize_embeddings2.size():
            raise ConfigurationError(
                "equalize_embeddings1 and equalize_embeddings2 must be the same size."
            )
        if equalize_embeddings1.ndim < 2:
            raise ConfigurationError(
                "equalize_embeddings1 and equalize_embeddings2 must have at least two dimensions."
            )
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError("evaluation_embeddings must have at least two dimensions.")
        if evaluation_embeddings.size()[1:] != equalize_embeddings1.size()[1:]:
            raise ConfigurationError(
                "evaluation_embeddings, equalize_embeddings1, and equalize_embeddings2 must have same size \
                except for 0th dim (i.e. batch dimension)."
            )
        if bias_direction.ndim != 1:
            raise ConfigurationError("bias_direction must be one-dimensional.")
        if evaluation_embeddings.size(-1) != bias_direction.size(-1):
            raise ConfigurationError(
                "All embeddings and bias_direction must have the same dimensionality."
            )

        with torch.set_grad_enabled(self.requires_grad):
            bias_direction = bias_direction / torch.linalg.norm(bias_direction)

            bias_mitigated_embeddings = self._remove_component(
                evaluation_embeddings, bias_direction, normalize=True
            )

            mean_equalize_embeddings = (equalize_embeddings1 + equalize_embeddings2) / 2
            y = self._remove_component(mean_equalize_embeddings, bias_direction, normalize=True)
            z = torch.sqrt(1 - torch.square(torch.linalg.norm(y, dim=-1, keepdim=True)))
            z = torch.where(
                torch.matmul(
                    equalize_embeddings1 - equalize_embeddings2, bias_direction.reshape(-1, 1)
                )
                < 0,
                -z,
                z,
            )
            return torch.cat(
                [bias_mitigated_embeddings, z * bias_direction + y, -z * bias_direction + y]
            )


class LinearBiasMitigator(BiasMitigator):
    """
    Linear bias mitigator. Mitigates bias in embeddings by removing component
    in the bias direction.

    !!! Note
        For a detailed walkthrough and visual descriptions of the steps, please
        refer to Figure 3 in [VERB: Visualizing and Interpreting Bias Mitigation Techniques
        for Word Representations](https://api.semanticscholar.org/CorpusID:233168618).

    Based on: S. Dev and J. M. Phillips. [Attenuating bias in word vectors]
    (https://api.semanticscholar.org/CorpusID:59158788).
    In International Conference on Artificial Intelligence and Statistics,
    Proceedings of Machine Learning Research, pages 879–887. PMLR, 2019.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(self, evaluation_embeddings: torch.Tensor, bias_direction: torch.Tensor):
        """

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        # Parameters

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) of embeddings for which to mitigate bias.
        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.

        !!! Note
            All tensors are expected to be on the same device.

        # Returns

        bias_mitigated_embeddings : `torch.Tensor`
            A tensor of the same size as evaluation_embeddings.
        """
        # Some sanity checks
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError("evaluation_embeddings must have at least two dimensions.")
        if bias_direction.ndim != 1:
            raise ConfigurationError("bias_direction must be one-dimensional.")
        if evaluation_embeddings.size(-1) != bias_direction.size(-1):
            raise ConfigurationError(
                "All embeddings and bias_direction must have the same dimensionality."
            )

        with torch.set_grad_enabled(self.requires_grad):
            bias_direction = bias_direction / torch.linalg.norm(bias_direction)
            return self._remove_component(evaluation_embeddings, bias_direction)


class INLPBiasMitigator(BiasMitigator):
    """
    Iterative Nullspace Projection. It mitigates bias by repeatedly building
    a linear classifier that separates concept groups and linearly
    projecting all words along the classifier normal.

    !!! Note
        For a detailed walkthrough and visual descriptions of the steps, please
        refer to Figure 5 in [VERB: Visualizing and Interpreting Bias Mitigation Techniques
        for Word Representations](https://api.semanticscholar.org/CorpusID:233168618).

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

    def __call__(
        self,
        evaluation_embeddings: torch.Tensor,
        seed_embeddings1: torch.Tensor,
        seed_embeddings2: torch.Tensor,
        num_iters: int = 35,
    ):
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (evaluation_batch_size, ..., dim) of embeddings for which to mitigate bias.
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
        num_iters: `torch.Tensor`
            Number of times to build classifier and project embeddings along normal.

        !!! Note
            seed_embeddings1 and seed_embeddings2 need NOT be the same size. Furthermore,
            the embeddings at the same positions in each of seed_embeddings1 and seed_embeddings2
            are NOT expected to form seed word pairs.

        !!! Note
            All tensors are expected to be on the same device.

        !!! Note
            This bias mitigator is not differentiable.

        # Returns

        bias_mitigated_embeddings : `torch.Tensor`
            A tensor of the same size as evaluation_embeddings.
        """
        # Some sanity checks
        if seed_embeddings1.ndim < 2 or seed_embeddings2.ndim < 2:
            raise ConfigurationError(
                "seed_embeddings1 and seed_embeddings2 must have at least two dimensions."
            )
        if seed_embeddings1.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError("All seed embeddings must have same dimensionality.")
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError("evaluation_embeddings must have at least two dimensions.")
        if evaluation_embeddings.size(-1) != seed_embeddings1.size(
            -1
        ) or evaluation_embeddings.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError(
                "evaluation_embeddings, seed_embeddings1, and seed_embeddings2 must have the same dimensionality."
            )

        device = seed_embeddings1.device
        seed_embeddings1 = seed_embeddings1.flatten(end_dim=-2).detach().cpu().numpy()
        seed_embeddings2 = seed_embeddings2.flatten(end_dim=-2).detach().cpu().numpy()
        X = np.vstack([seed_embeddings1, seed_embeddings2])
        Y = np.concatenate([[0] * seed_embeddings1.shape[0], [1] * seed_embeddings2.shape[0]])

        rowspace_projs = []
        for iter_idx in range(num_iters):
            classifier = sklearn.svm.SVC(kernel="linear").fit(X, Y)
            weights = np.expand_dims(classifier.coef_[0], 0)

            if (np.linalg.norm(weights) < 1e-10 or classifier.score(X, Y) < 0.55) and iter_idx > 1:
                break

            rowspace_projs.append(self._get_rowspace_proj(weights))
            # Project embeddings to intersection of nullspaces
            nullspace_proj = np.eye(seed_embeddings1.shape[1]) - self._get_rowspace_proj(
                np.sum(rowspace_projs, axis=0)
            )
            evaluation_embeddings = torch.matmul(
                evaluation_embeddings, torch.from_numpy(nullspace_proj).float().t().to(device)
            )
            X = nullspace_proj.dot(X.T).T

        return evaluation_embeddings

    def _get_rowspace_proj(self, weights: np.ndarray):
        # Compute orthogonal basis
        if np.allclose(weights, 0):
            weights_basis = np.zeros_like(weights.T)
        else:
            weights_basis = scipy.linalg.orth(weights.T)
        # Get rowspace projection
        return weights_basis.dot(weights_basis.T)


class OSCaRBiasMitigator(BiasMitigator):
    """
    OSCaR bias mitigator. Mitigates bias in embeddings by dissociating concept subspaces
    through subspace orthogonalization. Formally, OSCaR applies a graded rotation
    on the embedding space to rectify two ideally-independent concept subspaces
    so that they become orthogonal.

    !!! Note
        For a detailed walkthrough and visual descriptions of the steps, please
        refer to Figure 6 in [VERB: Visualizing and Interpreting Bias Mitigation Techniques
        for Word Representations](https://api.semanticscholar.org/CorpusID:233168618).

    Based on: Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [OSCaR: Orthogonal Subspace
    Correction and Rectification of Biases in Word Embeddings](https://api.semanticscholar.org/CorpusID:220281039).
    ArXiv, abs/2007.00049.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __call__(
        self,
        evaluation_embeddings: torch.Tensor,
        bias_direction1: torch.Tensor,
        bias_direction2: torch.Tensor,
    ):
        """

        # Parameters

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) of embeddings for which to mitigate bias.
        bias_direction1 : `torch.Tensor`
            A unit tensor of size (dim, ) representing a concept subspace (e.g. gender).
        bias_direction2 : `torch.Tensor`
            A unit tensor of size (dim, ) representing another concept subspace from
            which bias_direction1 should be dissociated (e.g. occupation).

        !!! Note
            All tensors are expected to be on the same device.

        # Returns

        bias_mitigated_embeddings : `torch.Tensor`
            A tensor of the same size as evaluation_embeddings.
        """
        # Some sanity checks
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError("evaluation_embeddings must have at least two dimensions.")
        if bias_direction1.ndim != 1 or bias_direction2.ndim != 1:
            raise ConfigurationError("bias_direction1 and bias_direction2 must be one-dimensional.")
        if evaluation_embeddings.size(-1) != bias_direction1.size(-1) or evaluation_embeddings.size(
            -1
        ) != bias_direction2.size(-1):
            raise ConfigurationError(
                "All embeddings, bias_direction1, and bias_direction2 must have the same dimensionality."
            )
        if bias_direction1.size(-1) < 2:
            raise ConfigurationError(
                "Dimensionality of all embeddings, bias_direction1, and bias_direction2 must \
                be >= 2."
            )

        with torch.set_grad_enabled(self.requires_grad):
            bias_direction1 = bias_direction1 / torch.linalg.norm(bias_direction1)
            bias_direction2 = bias_direction2 / torch.linalg.norm(bias_direction2)

            bias_direction2_orth = self._remove_component(
                bias_direction2.reshape(1, -1), bias_direction1
            )[0]
            bias_direction2_orth = bias_direction2_orth / torch.linalg.norm(bias_direction2_orth)

            # Create rotation matrix as orthonormal basis
            # with v1 and v2'
            init_orth_matrix = torch.eye(
                bias_direction1.size(0),
                device=evaluation_embeddings.device,
                requires_grad=self.requires_grad,
            )
            rotation_matrix = torch.zeros(
                (bias_direction1.size(0), bias_direction1.size(0)),
                device=evaluation_embeddings.device,
                requires_grad=self.requires_grad,
            )
            rotation_matrix = torch.cat(
                [
                    bias_direction1.reshape(1, -1),
                    bias_direction2_orth.reshape(1, -1),
                    rotation_matrix[2:],
                ]
            )
            # Apply Gram-Schmidt
            for i in range(len(rotation_matrix) - 2):
                subspace_proj = torch.sum(
                    self._proj(
                        rotation_matrix[: i + 2].clone(), init_orth_matrix[i], normalize=True
                    ),
                    dim=0,
                )
                rotation_matrix[i + 2] = (init_orth_matrix[i] - subspace_proj) / torch.linalg.norm(
                    init_orth_matrix[i] - subspace_proj
                )

            mask = ~(evaluation_embeddings == 0).all(dim=-1)
            # Transform all evaluation embeddings
            # using orthonormal basis computed above
            rotated_evaluation_embeddings = torch.matmul(
                evaluation_embeddings[mask], rotation_matrix.t()
            )
            # Want to adjust first 2 coordinates and leave d - 2
            # other orthogonal components fixed
            fixed_rotated_evaluation_embeddings = rotated_evaluation_embeddings[..., 2:]
            # Restrict attention to subspace S spanned by bias directions
            # which we hope to make orthogonal
            restricted_rotated_evaluation_embeddings = torch.cat(
                [
                    torch.matmul(rotated_evaluation_embeddings, bias_direction1.reshape(-1, 1)),
                    torch.matmul(
                        rotated_evaluation_embeddings, bias_direction2_orth.reshape(-1, 1)
                    ),
                ],
                dim=-1,
            )

            # Transform and restrict bias directions
            restricted_bias_direction1 = torch.tensor(
                [1.0, 0.0], device=evaluation_embeddings.device, requires_grad=self.requires_grad
            )
            bias_direction_inner_prod = torch.dot(bias_direction1, bias_direction2)
            restricted_bias_direction2 = torch.tensor(
                [
                    bias_direction_inner_prod,
                    torch.sqrt(1 - torch.square(bias_direction_inner_prod)),
                ],
                device=evaluation_embeddings.device,
                requires_grad=self.requires_grad,
            )
            restricted_bias_direction2_orth = torch.tensor(
                [0.0, 1.0], device=evaluation_embeddings.device, requires_grad=self.requires_grad
            )

            restricted_bias_direction_inner_prod = torch.dot(
                restricted_bias_direction1, restricted_bias_direction2
            )
            theta = torch.abs(torch.arccos(restricted_bias_direction_inner_prod))
            theta_proj = np.pi / 2 - theta
            phi = torch.arccos(
                torch.matmul(
                    restricted_rotated_evaluation_embeddings
                    / torch.linalg.norm(
                        restricted_rotated_evaluation_embeddings, dim=-1, keepdim=True
                    ),
                    restricted_bias_direction1,
                )
            )
            d = torch.matmul(
                restricted_rotated_evaluation_embeddings
                / torch.linalg.norm(restricted_rotated_evaluation_embeddings, dim=-1, keepdim=True),
                restricted_bias_direction2_orth,
            )

            # Add noise to avoid DivideByZero
            theta_x = torch.zeros_like(phi, requires_grad=self.requires_grad)
            theta_x = torch.where(
                (d > 0) & (phi < theta_proj),
                theta * (phi / (theta_proj + 1e-10)),
                theta_x,
            )
            theta_x = torch.where(
                (d > 0) & (phi > theta_proj),
                theta * ((np.pi - phi) / (np.pi - theta_proj + 1e-10)),
                theta_x,
            )
            theta_x = torch.where(
                (d < 0) & (phi >= np.pi - theta_proj),
                theta * ((np.pi - phi) / (theta_proj + 1e-10)),
                theta_x,
            )
            theta_x = torch.where(
                (d < 0) & (phi < np.pi - theta_proj),
                theta * (phi / (np.pi - theta_proj + 1e-10)),
                theta_x,
            )

            f_matrix = torch.cat(
                [
                    torch.cos(theta_x).unsqueeze(-1),
                    -torch.sin(theta_x).unsqueeze(-1),
                    torch.sin(theta_x).unsqueeze(-1),
                    torch.cos(theta_x).unsqueeze(-1),
                ],
                dim=-1,
            )
            f_matrix = f_matrix.reshape(f_matrix.size()[:-1] + (2, 2))

            evaluation_embeddings_clone = evaluation_embeddings.clone()
            evaluation_embeddings_clone[mask] = torch.cat(
                [
                    torch.bmm(
                        f_matrix,
                        restricted_rotated_evaluation_embeddings.unsqueeze(-1),
                    ).squeeze(-1),
                    fixed_rotated_evaluation_embeddings,
                ],
                dim=-1,
            )
            return torch.matmul(evaluation_embeddings_clone, rotation_matrix)
