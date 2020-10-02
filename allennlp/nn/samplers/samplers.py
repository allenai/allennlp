import torch

from allennlp.nn.samplers.sampler import Sampler
from allennlp.nn.util import min_value_of_dtype


@Sampler.register("multinomial")
class MultinomialSampler(Sampler):
    """
    Represents a sampler to choose calues from a multinomial distribution

    Registered as a `Sampler` with name "multinomial".
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.sum(torch.abs(parameter))


@Sampler.register("TopK")
class TopKSampler(Sampler):
    """
    Represents a Sampler which redistributes the probability mass function among
    the top `k` choices then selects from that subset
    `logits` is a tensor of log-probabilities to be selected from.
    `k` is the number of highest-probability options that the returned choice will be selected from
    `temperature` modules the probabilitis of the selected tokens. A `temperature` below 1.0 produces a 
    sharper probability distribution and a `temperature` above 1.0 produces a flatter probability
    distribution.

    Registered as a `Sampler` with name "TopK".
    """
    def __init__(self, k: int = 1, temperature: float = 1.0, filter_val: float = -float('inf')):
        assert k >= 1, f"k must be >= 1"
        self.k = k
        self.temperature = temperature or 1.0
        self.filter_val = min_value_of_dtype(torch.float)

    def __call__(self, logits: torch.Tensor, k: int = 1) -> torch.Tensor:
        min_threshold = torch.topk(logits, k)[0][..., -1]
        filtered_indices = logits < min_threshold
        logits[filtered_indices] = filter_val

        filtered_probabilites = torch.nn.functional.softmax(filtered_logits_descending, dim=-1)
        return torch.multinomial(filtered_probabilites, 1)

@Sampler.register("TopP")
class TopPSampler(Sampler):
    """
    Represents a Sampler which redistributes the probability mass function among
    the top choices with a cumulative probability of at least `p` then selects from that subset
    `p` if minimum cumulative probability of highest-probability options that the returned choice will be selected from
    `temperature` modules the probabilitis of the selected tokens. A `temperature` below 1.0 produces a 
    sharper probability distribution and a `temperature` above 1.0 produces a flatter probability
    distribution.

    Registered as a `Sampler` with name "TopK".
    """
    def __init__(self, p: float = 0.9, temperature: float = 1.0, filter_val: float = -1000):
        assert p <= 1.0, f"p must be <= 0"
        self.p = p
        self.temperature = temperature or 1.0
        self.filter_val = min_value_of_dtype(torch.float)
            


    def __call__(self, logits: torch.Tensor ) -> torch.Tensor:
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
        filtered_indices = probabilities_summed < self.p
        # We want to include the firt index where probabilities_summes >= p, so we shift over one
        filtered_indices[..., 1:] = filtered_indices[..., :-1].clone()
        filtered_indices[..., 0] = 0

        # Here we set the filtered indices in the original logits to be the filter value
        logits[sorting_indices[filtered_indices]] = self.filter_val
        #filtered_logits_descending = torch.where(filtered_indices, self.filter_val, logits_descending)
        

        filtered_probabilites = torch.nn.functional.softmax(logits, dim=-1)
        print(filtered_probabilites)

        return torch.multinomial(filtered_probabilites, 1)
