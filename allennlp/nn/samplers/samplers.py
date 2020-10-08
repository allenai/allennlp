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
        self, logits: torch.Tensor, num_samples: int = 1, with_replacement: bool = True
    ) -> torch.Tensor:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        selected_indices = torch.multinomial(
            probabilities, num_samples, replacement=with_replacement
        )

        return (torch.gather(logits, 1, selected_indices), selected_indices)


@Sampler.register("top-k")
class TopKSampler(Sampler):
    """
    Represents a `Sampler` which redistributes the probability mass function among
    the top `k` choices then selects from that subset
    `logits` is a tensor of log-probabilities to be selected from.
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
        self, logits: torch.Tensor, num_samples: int = 1, with_replacement: bool = True
    ) -> torch.Tensor:

        assert self.k <= len(logits)

        min_threshold = torch.topk(logits, self.k)[0][..., -1].unsqueeze(dim=-1)
        filtered_indices = logits < min_threshold
        logits[..., filtered_indices] = self.filter_val

        filtered_probabilites = torch.nn.functional.softmax(logits, dim=-1)
        selected_indices = torch.multinomial(
            filtered_probabilites, num_samples, replacement=with_replacement
        )

        # Return (selected log probabilities, selected classes)
        # shape: (len(logits),1) , (len(logits), 1)
        return (torch.gather(logits, 1, selected_indices), selected_indices)


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
        self, logits: torch.Tensor, num_samples: int = 1, with_replacement: bool = True
    ) -> torch.Tensor:
        """
        Performs top-p sampling on the given `logits`.
        `logits` is a tensor of log-probabilities to be selected from.
        Returns the
        """
        # First apply temperature coefficient:
        logits = logits / self.temperature

        # Sort the probabilities to highest-first, then find the cumulative sum accross those
        logits_descending, sorting_indices = torch.sort(logits, descending=True)
        probabilities_descending = torch.nn.functional.softmax(logits_descending, dim=-1)
        probabilities_summed = torch.cumsum(probabilities_descending, dim=-1)

        # When the cumulative sum reaches p, replace all remaining with `filter_val`
        filtered_indices = probabilities_summed >= self.p

        # We want to include the firt index where probabilities_summes >= p, so we shift over one
        filtered_indices[..., 1:] = filtered_indices[..., :-1].clone()
        filtered_indices[..., 0] = 0

        # print('rifght', sorting_indices[filtered_indices])
        # Here we set the filtered indices in the original logits to be the filter value
        logits[..., sorting_indices[filtered_indices]] = self.filter_val

        filtered_probabilites = torch.nn.functional.softmax(logits, dim=-1)

        # Here we sample from the filtered distribution
        selected_indices = torch.multinomial(
            filtered_probabilites, num_samples, replacement=with_replacement
        )

        # Return (selected log probabilities, selected classes)
        # shape: (len(logits),1) , (len(logits), 1)
        return (torch.gather(logits, 1, selected_indices), selected_indices)
