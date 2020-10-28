import torch

from allennlp.nn.samplers.sampler import Sampler
from allennlp.nn.util import min_value_of_dtype


@Sampler.register("multinomial")
class MultinomialSampler(Sampler):
    """
    Represents a sampler to choose values from a multinomial distribution.

    Registered as a `Sampler` with name "multinomial".
    """

    def __init__(self, temperature: float = 1.0, filter_val: float = -float("inf")) -> None:
        self.temperature = temperature
        self.filter_val = min_value_of_dtype(torch.float)

    def __call__(
        self,
        log_probs: torch.Tensor,
        perturbed_log_probs: torch.Tensor = None,
        num_samples: int = 1,
        with_replacement: bool = True,
    ) -> torch.Tensor:
        probabilities = log_probs.exp()

        selected_indices = torch.multinomial(
            probabilities, num_samples, replacement=with_replacement
        )

        return (torch.gather(log_probs, 1, selected_indices), selected_indices)


@Sampler.register("top-k")
class TopKSampler(Sampler):
    """
    Represents a `Sampler` which redistributes the probability mass function among
    the top `k` choices then selects from that subset
    `log_probs` is a tensor of log-probabilities to be selected from.
    `k` is the number of highest-probability options that the returned choice will be selected from
    `temperature` modules the probabilitis of the selected tokens. A `temperature` below 1.0 produces a
    sharper probability distribution and a `temperature` above 1.0 produces a flatter probability
    distribution.

    Registered as a `Sampler` with name "top-k".
    """

    def __init__(self, k: int = 1, temperature: float = 1.0, filter_val: float = -float("inf")):
        assert k >= 1, f'{"k must be >= 1"}'
        self.k = k
        self.temperature = temperature or 1.0
        self.filter_val = min_value_of_dtype(torch.float)

    def __call__(
        self,
        log_probs: torch.Tensor,
        perturbed_log_probs: torch.Tensor = None,
        num_samples: int = 1,
        with_replacement: bool = True,
    ) -> torch.Tensor:

        assert self.k <= len(log_probs)

        if perturbed_log_probs is None:
            perturbed_log_probs = log_probs.clone()

        # First apply temperature coefficient:
        perturbed_log_probs = perturbed_log_probs / self.temperature

        # Find the indices that are not to be selected from
        min_threshold = torch.topk(perturbed_log_probs, self.k)[0][..., -1].unsqueeze(dim=-1)
        filtered_indices = log_probs < min_threshold

        # Prevent the filtered indiceds from being selected
        log_probs[..., filtered_indices] = self.filter_val
        filtered_probabilites = torch.nn.functional.softmax(log_probs, dim=-1)

        # Sample from the remaining indices
        selected_indices = torch.multinomial(
            filtered_probabilites, num_samples, replacement=with_replacement
        )

        # Return (selected log probabilities, selected classes)
        # shape: (len(log_probs),1) , (len(log_probs), 1)
        return (torch.gather(perturbed_log_probs, 1, selected_indices), selected_indices)


@Sampler.register("top-p")
class TopPSampler(Sampler):
    """
    Represents a `Sampler` which redistributes the probability mass function among
    the top choices with a cumulative probability of at least `p` then selects from that subset
    `p` if minimum cumulative probability of highest-probability options that the returned
    choice will be selected from `temperature` modules the probabilitis of the selected tokens.
    A `temperature` below 1.0 produces a sharper probability distribution and a `temperature`
    above 1.0 produces a flatter probability distribution.

    Registered as a `Sampler` with name "top-p".
    """

    def __init__(self, p: float = 0.9, temperature: float = 1.0, filter_val: float = -float("inf")):
        assert p <= 1.0, f'{"p must be <= 0"}'
        self.p = p
        self.temperature = temperature or 1.0
        self.filter_val = min_value_of_dtype(torch.float)

    def __call__(
        self,
        log_probs: torch.Tensor,
        perturbed_log_probs: torch.Tensor = None,
        num_samples: int = 1,
        with_replacement: bool = True,
    ) -> torch.Tensor:
        """
        Performs top-p sampling on the given `log_probs`.
        `log_probs` is a tensor of log-probabilities to be selected from.
        Returns the
        """
        if perturbed_log_probs is None:
            perturbed_log_probs = log_probs.clone()

        # First apply temperature coefficient:
        perturbed_log_probs = perturbed_log_probs / self.temperature

        # Sort the probabilities to highest-first, then find the cumulative sum accross those
        log_probs_descending, sorting_indices = torch.sort(perturbed_log_probs, descending=True)
        probabilities_descending = torch.nn.functional.softmax(log_probs_descending, dim=-1)
        probabilities_summed = torch.cumsum(probabilities_descending, dim=-1)

        # When the cumulative sum reaches p, replace all remaining with `filter_val`
        filtered_indices = probabilities_summed >= self.p

        # We want to include the firt index where probabilities_summes >= p, so we shift over one
        filtered_indices[..., 1:] = filtered_indices[..., :-1].clone()
        filtered_indices[..., 0] = 0

        for row, sort_indices, filter_vals in zip(
            perturbed_log_probs, sorting_indices, filtered_indices
        ):
            filter_idx = sort_indices[filter_vals]
            row[filter_idx] = self.filter_val

        filtered_probabilites = torch.nn.functional.softmax(perturbed_log_probs, dim=-1)

        # Here we sample from the filtered distribution
        selected_indices = torch.multinomial(
            filtered_probabilites, num_samples, replacement=with_replacement
        )

        # Return (selected log probabilities, selected classes)
        # shape: (len(log_probs),1) , (len(log_probs), 1)
        return (torch.gather(log_probs, 1, selected_indices), selected_indices)


@Sampler.register("gumbel-max")
class GumbelMaxSampler(Sampler):
    """
    Represents a `Sampler` which uses the Gumbel-Max trick to sample `num_samples`
    instances without replacement, See
    [*Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling
    Sequences Without Replacement*, W Kool, H Van Hoof and M Welling, 2010]
    (https://arxiv.org/abs/1903.06059).
    `log_probs` is a tensor of normalized log-probabilities of the partial sequence up to that point
    `T` is a tensor of the previous sampled perturbed log-probabilities
    `num_samples` is the number of instances to sample

    Returns a tensor of the top `num_samples` perturbed log probabilities and their
    indices within `log_probs`.

    Registered as a `Sampler` with name "gumbel-max".
    """

    def __init__(self):
        self.num_samples = 1

    def __call__(
        self,
        log_probs: torch.Tensor,
        perturbed_log_probs: torch.Tensor = None,
        num_samples: int = 1,
        with_replacement: bool = True,
    ) -> torch.Tensor:
        # Make sure we're not trying to select more than available
        assert num_samples <= log_probs.size(-1)

        if perturbed_log_probs is None:
            perturbed_log_probs = torch.zeros(len(log_probs), 1)

        # Next, we find the gumbel perturbed probabilites by sampling elements
        # from the gumbel distribution and shifting them to `log_probs`
        g_phi = self._gumbel(log_probs) + log_probs

        # Now we find the maximum from these samples
        Z, _ = g_phi.max(dim=-1)

        # Find the truncated gumbel distribution conditioned on max = Z
        # (numerically stable implementation)
        g_phi_tilde = self._truncated_gumbel(perturbed_log_probs, Z, g_phi)

        # Select the top (max, argmax) instances from the truncated Gumbel
        return torch.topk(g_phi_tilde, num_samples)

    def _gumbel(self, log_probs):
        return -torch.log(-torch.log(torch.rand_like(log_probs)))

    def _truncated_gumbel(self, perturbed_log_probs, Z, g_phi):
        # Computation of the truncatd gumbel distribution can be computationally
        # unstable. Instead, we compute:
        v = perturbed_log_probs - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim=-1)))
        g_phi_tilde = (
            perturbed_log_probs - torch.nn.functional.relu(v) - torch.log1p(torch.exp(-v.abs()))
        )
        return g_phi_tilde
