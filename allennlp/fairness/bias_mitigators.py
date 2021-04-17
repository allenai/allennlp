import torch
import numpy as np
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

    def _proj(u: torch.Tensor, v: torch.Tensor, normalize: bool = False):
        proj = torch.matmul(u, v.reshape(-1, 1)) * v
        if normalize:
            return proj / torch.dot(v, v)
        return proj

    def _remove_component(
        embeddings: torch.Tensor, bias_direction: torch.Tensor, normalize: bool = False
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

    Description taken from: Goenka, D. (2020). [Tackling Gender Bias in Word Embeddings]
    (https://towardsdatascience.com/tackling-gender-bias-in-word-embeddings-c965f4076a10).

    Implementation based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
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
            bias_direction /= torch.linalg.norm(bias_direction)

            bias_mitigated_embeddings = self._remove_component(
                evaluation_embeddings, bias_direction, normalize=True
            )

            mean_equalize_embeddings = (equalize_embeddings1 + equalize_embeddings2) / 2
            y = self._remove_component(mean_equalize_embeddings, bias_direction, normalize=True)
            z = torch.sqrt(1 - torch.square(torch.linalg.norm(y, dim=-1, keepdim=True)))
            z[
                torch.matmul(
                    equalize_embeddings1 - equalize_embeddings2, bias_direction.reshape(-1, 1)
                )
                < 0
            ] *= -1
            return torch.cat(
                [bias_mitigated_embeddings, z * bias_direction + y, -z * bias_direction + y]
            )


class LinearBiasMitigator(BiasMitigator):
    """
    Linear bias mitigator. Mitigates bias in embeddings by removing component
    in the bias direction.

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
            bias_direction /= torch.linalg.norm(bias_direction)
            return self._remove_component(evaluation_embeddings, bias_direction)


class OSCaRBiasMitigator(BiasMitigator):
    """
    OSCaR bias mitigator. Mitigates bias in embeddings by dissociating concept subspaces
    through subspace orthogonalization. Formally, OSCaR applies a graded rotation
    on the embedding space to rectify two ideally-independent concept subspaces
    so that they become orthogonal.

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
            bias_direction1 /= torch.linalg.norm(bias_direction1)
            bias_direction2 /= torch.linalg.norm(bias_direction2)

            bias_direction2_orth = self._remove_component(
                bias_direction2.reshape(1, -1), bias_direction1
            )[0]
            bias_direction2_orth /= torch.linalg.norm(bias_direction2_orth)

            # Create rotation matrix as orthonormal basis
            # with v1 and v2'
            init_orth_matrix = torch.eye(
                bias_direction1.size(0), device=evaluation_embeddings.device
            )
            rotation_matrix = torch.zeros(
                bias_direction1.size(0), device=evaluation_embeddings.device
            )
            rotation_matrix[0] = bias_direction1
            rotation_matrix[1] = bias_direction2_orth
            # Apply Gram-Schmidt
            for i in range(len(rotation_matrix) - 2):
                subspace_proj = torch.sum(
                    self._proj(rotation_matrix[: i + 2], init_orth_matrix[i], normalize=True), dim=0
                )
                rotation_matrix[i + 2] = init_orth_matrix[i] - subspace_proj
                rotation_matrix[i + 2] /= torch.linalg.norm(rotation_matrix[i + 2])

            mask = torch.count_nonzero(evaluation_embeddings, dim=-1) != 0
            # Transform all evaluation embeddings
            # using orthonormal basis computed above
            rotated_evaluation_embeddings = torch.matmul(
                rotation_matrix, evaluation_embeddings[mask]
            )
            # Want to adjust first 2 coordinates and leave d - 2
            # other orthogonal components fixed
            fixed_rotated_evaluation_embeddings = rotated_evaluation_embeddings[..., 2:]
            # Restrict attention to subspace S
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
            restricted_bias_direction1 = torch.Tensor([1.0, 0.0])
            bias_direction_inner_prod = torch.dot(bias_direction1, bias_direction2)
            restricted_bias_direction2 = torch.Tensor(
                [bias_direction_inner_prod, torch.sqrt(1 - torch.square(bias_direction_inner_prod))]
            )
            restricted_bias_direction2_orth = torch.Tensor([0.0, 1.0])

            restricted_bias_direction_inner_prod = torch.dot(
                restricted_bias_direction1, restricted_bias_direction2
            )
            theta = abs(torch.arccos(restricted_bias_direction_inner_prod).item())
            theta_proj = np.pi / 2 - theta
            phi = torch.arccos(
                torch.matmul(
                    restricted_rotated_evaluation_embeddings
                    / torch.linalg.norm(restricted_rotated_evaluation_embeddings, dim=-1),
                    restricted_bias_direction1,
                )
            )
            d = torch.matmul(
                restricted_rotated_evaluation_embeddings
                / torch.linalg.norm(restricted_rotated_evaluation_embeddings, dim=-1),
                restricted_bias_direction2_orth,
            )

            # Add noise to avoid DivideByZero
            theta_x = torch.zeros_like(phi)
            theta_x[(d > 0) & (phi < theta_proj)] = theta * (
                phi[(d > 0) & (phi < theta_proj)] / (theta_proj + 1e-10)
            )
            theta_x[(d > 0) & (phi > theta_proj)] = theta * (
                (np.pi - phi[(d > 0) & (phi > theta_proj)]) / (np.pi - theta_proj + 1e-10)
            )
            theta_x[(d < 0) & (phi >= np.pi - theta_proj)] = theta * (
                phi[(d < 0) & (phi >= np.pi - theta_proj)] / (theta_proj + 1e-10)
            )
            theta_x[(d < 0) & (phi < np.pi - theta_proj)] = theta * (
                phi[(d < 0) & (phi < np.pi - theta_proj)] / (np.pi - theta_proj + 1e-6)
            )

            f_matrix = torch.zeros(theta_x.size() + (4), device=theta_x.device)
            f_matrix[..., 0] = torch.cos(theta_x)
            f_matrix[..., 1] = -torch.sin(theta_x)
            f_matrix[..., 2] = torch.sin(theta_x)
            f_matrix[..., 3] = torch.cos(theta_x)
            f_matrix = f_matrix.reshape(f_matrix.size()[:-1] + (2, 2))

            evaluation_embeddings[mask] = torch.cat(
                [
                    torch.matmul(f_matrix, restricted_rotated_evaluation_embeddings),
                    fixed_rotated_evaluation_embeddings,
                ],
                dim=-1,
            )
            return torch.matmul(rotation_matrix.transpose(0, 1), evaluation_embeddings)
