import torch


class BiasDirection(torch.nn.Module):
    """
    Bias direction. Computes one-dimensional subspace that is the span
    of a specific concept (e.g. gender) by a variety of subspace
    identification methods, e.g. PCA, Paired-PCA, 2-Means, Classification
    Normal. It is uncommon to utilize more than one direction to
    represent a concept.

    Implementation and terminology based on Rathore, A., Dev, S., Phillips, J.M., Srikumar,
    V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021).
    [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
    Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
    ArXiv, abs/2104.02797.
    """

    def __init__(self, bias_direction_method: str = "pca", requires_grad: bool = False):
        """

        # Parameters

        bias_direction_method : `str`, optional (default=`"pca"`)
            Method by which to compute a one-dimensional subspace that is the span
            of a specific concept (e.g. gender). Currently-implemented bias direction
            methods: `"pca"`, `"paired_pca"`, `"two_means"`, `"classification_normal"`, `"svd"`.
        requires_grad : `bool`, optional (default=`False`)
            Option to enable gradient calculation.
        """
        self.bias_direction_method = getattr(self, "_" + bias_direction_method, None)
        if self.bias_direction_method is None:
            raise NotImplementedError(
                "The bias direction method {} has not been implemented yet.".format(
                    bias_direction_method
                )
            )
        self.requires_grad = requires_grad

    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor = None):
        """

        # Parameters

        embeddings1 : `torch.Tensor`
            A tensor of size (batch_size, ..., dim) containing a set of embeddings
            for the first-words from pairs of seed words related to a concept (e.g. gender).
        embeddings2: `torch.Tensor`, optional (default=`None`)
            A tensor of size (batch_size, ..., dim) containing a set of embeddings
            for the second-words from pairs of seed words related to a concept (e.g. gender).
            embeddings2 is required for every bias direction method except vanilla PCA;
            vanilla PCA disregards embeddings2.

        !!! Note
            With the exception of for PCA, the embeddings at the same position in each of embeddings1
            and embeddings2 are expected to form seed word pairs from which the bias direction can be
            extrapolated. For example, if the concept is gender, the embeddings for
            ("man", "woman") should be at the same positions in embeddings1 and embeddings2.
            Importantly, this is a binary treatment of gender identity and does not accurately
            characterize gender in real life.
        """
        with torch.set_grad_enabled(self.requires_grad):
            self.bias_direction_method(embeddings1, embeddings2)

    def _pca(self, embeddings1, embeddings2):
        # pca_lowrank centers the embeddings by default
        _, _, V = torch.pca_lowrank(embeddings1, q=2)
        # get top principal component
        bias_direction = V[0]
        return self._normalize_bias_direction(bias_direction)

    def _paired_pca(self, embeddings1, embeddings2):
        paired_embeddings = embeddings1 - embeddings2
        _, _, V = torch.pca_lowrank(paired_embeddings, q=2)
        bias_direction = V[0]
        return self._normalize_bias_direction(bias_direction)

    def _two_means(self, embeddings1, embeddings2):
        embeddings1_mean, embeddings2_mean = torch.mean(embeddings1, dim=0), torch.mean(
            embeddings2, dim=0
        )
        bias_direction = (embeddings1_mean - embeddings2_mean) / torch.linalg.norm(
            embeddings1_mean - embeddings2_mean
        )
        return self._normalize_bias_direction(bias_direction)

    def _normalize_bias_direction(cls, bias_direction):
        return bias_direction / torch.linalg.norm(bias_direction)