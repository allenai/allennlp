import torch
from typing import Union
from os import PathLike

from allennlp.fairness import (
    HardBiasMitigator,
    LinearBiasMitigator,
    INLPBiasMitigator,
    OSCaRBiasMitigator,
)
from allennlp.fairness.bias_direction_wrappers import BiasDirectionWrapper
from allennlp.fairness.bias_utils import load_word_pairs

from allennlp.common import Registrable
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary


class BiasMitigatorWrapper(Registrable):
    """
    Parent class for bias mitigator wrappers.
    """

    def train(self, mode: bool = True):
        """

        # Parameters

        mode : `bool`, optional (default=`True`)
            Sets `requires_grad` to value of `mode` for bias mitigator
            and associated bias direction.
        """
        raise NotImplementedError


# TODO: remove equalize words from evaluation words
@BiasMitigatorWrapper.register("hard")
class HardBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    equalize_word_pairs_file : `Union[PathLike, str]`
        Path of file containing equalize word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize equalize words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(
        self,
        bias_direction: BiasDirectionWrapper,
        equalize_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        mitigator_vocab: Vocabulary = None,
        namespace: str = "tokens",
        requires_grad: bool = True,
    ):
        self.bias_direction = bias_direction
        self.ids1, self.ids2 = load_word_pairs(
            equalize_word_pairs_file, tokenizer, mitigator_vocab, namespace
        )
        self.mitigator = HardBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        """
        Called as forward hook.
        """
        # embed subword token IDs and mean pool to get
        # embedding of original word
        ids1_embeddings = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(
                torch.mean(module.forward(i), dim=0, keepdim=True)
            )  # forward() does not trigger hooks, thereby avoiding infinite recursion
        ids2_embeddings = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings = torch.cat(ids1_embeddings)
        ids2_embeddings = torch.cat(ids2_embeddings)

        module_out_size = module_out.size()
        # flatten tensor except for last dimension
        module_out = module_out.flatten(end_dim=-2)
        # only return bias-mitigated evaluation embeddings
        module_out = self.mitigator(
            module_out, self.bias_direction(module), ids1_embeddings, ids2_embeddings
        )[: module_out.size(0)]
        return module_out.reshape(module_out_size)

    def train(self, mode: bool = True):
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)


@BiasMitigatorWrapper.register("linear")
class LinearBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(self, bias_direction: BiasDirectionWrapper, requires_grad: bool = True):
        self.bias_direction = bias_direction
        self.mitigator = LinearBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        """
        Called as forward hook.
        """
        module_out_size = module_out.size()
        # flatten tensor except for last dimension
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, self.bias_direction(module))
        return module_out.reshape(module_out_size)

    def train(self, mode: bool = True):
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)


@BiasMitigatorWrapper.register("inlp")
class INLPBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    """

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        mitigator_vocab: Vocabulary = None,
        namespace: str = "tokens",
    ):
        self.ids1, self.ids2 = load_word_pairs(
            seed_word_pairs_file, tokenizer, mitigator_vocab, namespace
        )
        self.mitigator = INLPBiasMitigator()

    def __call__(self, module, module_in, module_out):
        """
        Called as forward hook.
        """
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

        module_out_size = module_out.size()
        # flatten tensor except for last dimension
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(module_out, ids1_embeddings, ids2_embeddings)
        return module_out.reshape(module_out_size)

    def train(self, mode: bool = True):
        pass


@BiasMitigatorWrapper.register("oscar")
class OSCaRBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction1 : `BiasDirectionWrapper`
        Bias direction of first concept subspace used by mitigator.
    bias_direction2 : `BiasDirectionWrapper`
        Bias direction of second concept subspace used by mitigator.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(
        self,
        bias_direction1: BiasDirectionWrapper,
        bias_direction2: BiasDirectionWrapper,
        requires_grad: bool = True,
    ):
        self.bias_direction1 = bias_direction1
        self.bias_direction2 = bias_direction2
        self.mitigator = OSCaRBiasMitigator(requires_grad=requires_grad)

    def __call__(self, module, module_in, module_out):
        """
        Called as forward hook.
        """
        module_out_size = module_out.size()
        # flatten tensor except for last dimension
        module_out = module_out.flatten(end_dim=-2)
        module_out = self.mitigator(
            module_out, self.bias_direction1(module), self.bias_direction2(module)
        )
        return module_out.reshape(module_out_size)

    def train(self, mode: bool = True):
        self.mitigator.requires_grad = mode
        self.bias_direction1.train(mode)
        self.bias_direction2.train(mode)
