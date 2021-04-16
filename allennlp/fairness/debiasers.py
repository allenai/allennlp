import torch
from allennlp.common.checks import ConfigurationError


class Debiaser:
    """
    Parent class for bias debiaser classes.

    # Parameters

    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation.
    """

    def __init__(self, requires_grad: bool = False):
        self.requires_grad = requires_grad


class HardDebiaser(Debiaser):
    """
    Hard debiaser. Debiases embeddings by:

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

    def forward(
        self,
        evaluation_embeddings: torch.Tensor,
        bias_direction: torch.Tensor,
        equalize_embeddings1: torch.Tensor,
        equalize_embeddings2: torch.Tensor,
    ):
        """

        # Parameters

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (evaluation_batch_size, ..., dim) of embeddings to debias.
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

        # Returns

        debiased_embeddings : `torch.Tensor`
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
            debiased_embeddings = self._remove_component(evaluation_embeddings, bias_direction)

            mean_equalize_embeddings = (equalize_embeddings1 + equalize_embeddings2) / 2
            y = self._remove_component(mean_equalize_embeddings, bias_direction)
            z = torch.sqrt(1 - torch.square(torch.linalg.norm(y)))
            z[torch.matmul(equalize_embeddings1 - equalize_embeddings2, bias_direction) < 0] *= -1
            return torch.cat([debiased_embeddings, z * bias_direction + y, -z * bias_direction + y])

    def _remove_component(embeddings: torch.Tensor, bias_direction: torch.Tensor):
        return embeddings - torch.matmul(
            embeddings, bias_direction.reshape(-1, 1)
        ) * bias_direction / torch.dot(bias_direction, bias_direction)


class LinearDebiaser(Debiaser):
    """
    Linear debiaser. Debiases embeddings by removing component
    in the bias direction.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.

    # Parameters

    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation.
    """

    def forward(self, evaluation_embeddings: torch.Tensor, bias_direction: torch.Tensor):
        """

        # Parameters

        evaluation_embeddings : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) of embeddings to debias.
        bias_direction : `torch.Tensor`
            A unit tensor of size (dim, ) representing the concept subspace.

        # Returns

        debiased_embeddings : `torch.Tensor`
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
            return (
                evaluation_embeddings
                - torch.matmul(evaluation_embeddings, bias_direction.reshape(-1, 1))
                * bias_direction
            )

class Oscar(Debiaser):
    