import torch
from typing import Union, Optional
from os import PathLike

from allennlp.fairness.bias_direction import (
    BiasDirection,
    PCABiasDirection,
    PairedPCABiasDirection,
    TwoMeansBiasDirection,
    ClassificationNormalBiasDirection,
)
from allennlp.fairness.bias_utils import load_word_pairs, load_words

from allennlp.common import Registrable
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary


class BiasDirectionWrapper(Registrable):
    """
    Parent class for bias direction wrappers.
    """

    def __init__(self):
        self.direction: BiasDirection = None
        self.noise: float = None

    def __call__(self, module):
        raise NotImplementedError

    def train(self, mode: bool = True):
        """

        # Parameters

        mode : `bool`, optional (default=`True`)
            Sets `requires_grad` to value of `mode` for bias direction.
        """
        self.direction.requires_grad = mode

    def add_noise(self, t: torch.Tensor):
        """

        # Parameters

        t : `torch.Tensor`
            Tensor to which to add small amount of Gaussian noise.
        """
        return t + self.noise * torch.randn(t.size(), device=t.device)


@BiasDirectionWrapper.register("pca")
class PCABiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_words_file : `Union[PathLike, str]`
        Path of file containing seed words.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """

    def __init__(
        self,
        seed_words_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = "tokens",
        requires_grad: bool = False,
        noise: float = 1e-10,
    ):
        self.ids = load_words(seed_words_file, tokenizer, direction_vocab, namespace)
        self.direction = PCABiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module):
        # embed subword token IDs and mean pool to get
        # embedding of original word
        ids_embeddings = []
        for i in self.ids:
            i = i.to(module.weight.device)
            ids_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids_embeddings = torch.cat(ids_embeddings)

        # adding trivial amount of noise
        # to eliminate linear dependence amongst all embeddings
        # when training first starts
        ids_embeddings = self.add_noise(ids_embeddings)

        return self.direction(ids_embeddings)


@BiasDirectionWrapper.register("paired_pca")
class PairedPCABiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = "tokens",
        requires_grad: bool = False,
        noise: float = 1e-10,
    ):
        self.ids1, self.ids2 = load_word_pairs(
            seed_word_pairs_file, tokenizer, direction_vocab, namespace
        )
        self.direction = PairedPCABiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module):
        # embed subword token IDs and mean pool to get
        # embedding of original word
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)

        ids1_embeddings = self.add_noise(ids1_embeddings)
        ids2_embeddings = self.add_noise(ids2_embeddings)

        return self.direction(ids1_embeddings, ids2_embeddings)


@BiasDirectionWrapper.register("two_means")
class TwoMeansBiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = "tokens",
        requires_grad: bool = False,
        noise: float = 1e-10,
    ):
        self.ids1, self.ids2 = load_word_pairs(
            seed_word_pairs_file, tokenizer, direction_vocab, namespace
        )
        self.direction = TwoMeansBiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module):
        # embed subword token IDs and mean pool to get
        # embedding of original word
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)

        ids1_embeddings = self.add_noise(ids1_embeddings)
        ids2_embeddings = self.add_noise(ids2_embeddings)

        return self.direction(ids1_embeddings, ids2_embeddings)


@BiasDirectionWrapper.register("classification_normal")
class ClassificationNormalBiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = "tokens",
        noise: float = 1e-10,
    ):
        self.ids1, self.ids2 = load_word_pairs(
            seed_word_pairs_file, tokenizer, direction_vocab, namespace
        )
        self.direction = ClassificationNormalBiasDirection()
        self.noise = noise

    def __call__(self, module):
        # embed subword token IDs and mean pool to get
        # embedding of original word
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)

        ids1_embeddings = self.add_noise(ids1_embeddings)
        ids2_embeddings = self.add_noise(ids2_embeddings)

        return self.direction(ids1_embeddings, ids2_embeddings)
