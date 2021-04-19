import logging
from typing import List, Optional, Tuple, Union, Sequence

import numpy as np
from overrides import overrides
import torch
import torch.autograd as autograd

from allennlp.common import Lazy
from allennlp.common.tqdm import Tqdm
from allennlp.data import DatasetReader, DatasetReaderInput, Instance
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.models.model import Model


logger = logging.getLogger(__name__)


@InfluenceInterpreter.register("simple-influence")
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as an `InfluenceInterpreter` with name "simple-influence".

    This goes through every example in the train set to calculate the influence score. It uses
    [LiSSA (Linear time Stochastic Second-Order Algorithm)](https://api.semanticscholar.org/CorpusID:10569090)
    to approximate the inverse of the Hessian used for the influence score calculation.

    # Parameters

    lissa_batch_size : `int`, optional (default = `8`)
        The batch size to use for LiSSA.
        According to [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974),
        it is better to use batched samples for approximation for better stability.

    damping : `float`, optional (default = `3e-3`)
        This is a hyperparameter for LiSSA.
        A damping termed added in case the approximated Hessian (during LiSSA) has
        negative eigenvalues.

    num_samples : `int`, optional (default = `1`)
        This is a hyperparameter for LiSSA that we
        determine how many rounds of the recursion process we would like to run for approxmation.

    recursion_depth : `Union[float, int]`, optional (default = `0.25`)
        This is a hyperparameter for LiSSA that
        determines the recursion depth we would like to go through.
        If a `float`, it means X% of the training examples.
        If an `int`, it means recurse for X times.

    scale : `float`, optional, (default = `1e4`)
        This is a hyperparameter for LiSSA to tune such that the Taylor expansion converges.
        It is applied to scale down the loss during LiSSA to ensure that `H <= I`,
        where `H` is the Hessian and `I` is the identity matrix.

        See footnote 2 of [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974).

    !!! Note
        We choose the same default values for the LiSSA hyperparameters as
        [Han, Xiaochuang et al. (2020)](https://api.semanticscholar.org/CorpusID:218628619).
    """

    def __init__(
        self,
        model: Model,
        train_data_path: DatasetReaderInput,
        train_dataset_reader: DatasetReader,
        *,
        test_dataset_reader: Optional[DatasetReader] = None,
        train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        params_to_freeze: List[str] = None,
        cuda_device: int = -1,
        lissa_batch_size: int = 8,
        damping: float = 3e-3,
        num_samples: int = 1,
        recursion_depth: Union[float, int] = 0.25,
        scale: float = 1e4,
    ) -> None:
        super().__init__(
            model=model,
            train_data_path=train_data_path,
            train_dataset_reader=train_dataset_reader,
            test_dataset_reader=test_dataset_reader,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            params_to_freeze=params_to_freeze,
            cuda_device=cuda_device,
        )

        self._lissa_dataloader = SimpleDataLoader(
            list(self._train_loader.iter_instances()),
            lissa_batch_size,
            shuffle=True,
            vocab=self.vocab,
        )
        self._lissa_dataloader.set_target_device(self.device)
        if isinstance(recursion_depth, float) and recursion_depth > 0.0:
            self._lissa_dataloader.batches_per_epoch = int(
                len(self._lissa_dataloader) * recursion_depth
            )
        elif isinstance(recursion_depth, int) and recursion_depth > 0:
            self._lissa_dataloader.batches_per_epoch = recursion_depth
        else:
            raise ValueError("'recursion_depth' should be a positive int or float")

        self._damping = damping
        self._num_samples = num_samples
        self._recursion_depth = recursion_depth
        self._scale = scale

    @overrides
    def _calculate_influence_scores(
        self, test_instance: Instance, test_loss: float, test_grads: Sequence[torch.Tensor]
    ) -> List[float]:
        # Approximate the inverse of Hessian-Vector Product through LiSSA
        inv_hvp = get_inverse_hvp_lissa(
            test_grads,
            self.model,
            self.used_params,
            self._lissa_dataloader,
            self._damping,
            self._num_samples,
            self._scale,
        )
        return [
            # dL_test * d theta as in 2.2 of [https://arxiv.org/pdf/2005.06676.pdf]
            # TODO (epwalsh): should we divide `x.grads` by `self._scale`?
            torch.dot(inv_hvp, _flatten_tensors(x.grads)).item()
            for x in Tqdm.tqdm(self.train_instances, desc="scoring train instances")
        ]


def get_inverse_hvp_lissa(
    vs: Sequence[torch.Tensor],
    model: Model,
    used_params: Sequence[torch.Tensor],
    lissa_data_loader: DataLoader,
    damping: float,
    num_samples: int,
    scale: float,
) -> torch.Tensor:
    """
    This function approximates the product of the inverse of the Hessian and
    the vectors `vs` using LiSSA.

    Adapted from [github.com/kohpangwei/influence-release]
    (https://github.com/kohpangwei/influence-release/blob/0f656964867da6ddcca16c14b3e4f0eef38a7472/influence/genericNeuralNet.py#L475),
    the repo for [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974),
    and [github.com/xhan77/influence-function-analysis]
    (https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336),
    the repo for [Han, Xiaochuang et al. (2020)](https://api.semanticscholar.org/CorpusID:218628619).
    """
    inverse_hvps = [torch.tensor(0) for _ in vs]
    for _ in Tqdm.tqdm(range(num_samples), desc="LiSSA samples", total=num_samples):
        # See a explanation at "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
        # initialize \tilde{H}^{−1}_0 v = v
        cur_estimates = vs
        recursion_iter = Tqdm.tqdm(
            lissa_data_loader, desc="LiSSA depth", total=len(lissa_data_loader)
        )
        for j, training_batch in enumerate(recursion_iter):
            # TODO (epwalsh): should we make sure `model` is in "train" or "eval" mode here?
            model.zero_grad()
            train_output_dict = model(**training_batch)
            # Hessian of loss @ \tilde{H}^{−1}_{j - 1} v
            hvps = get_hvp(train_output_dict["loss"], used_params, cur_estimates)

            # This is the recursive step:
            # cur_estimate = \tilde{H}^{−1}_{j - 1} v
            # (i.e. Hessian-Vector Product estimate from last iteration)
            # Updating for \tilde{H}^{−1}_j v, the new current estimate becomes:
            # v + (I - (Hessian_at_x + damping)) * cur_estimate
            # = v + (I + damping) * cur_estimate - Hessian_at_x * cur_estimate
            # We divide `hvp / scale` here (or, equivalently `Hessian_at_x / scale`)
            # so that we're effectively dividing the loss by `scale`.
            cur_estimates = [
                v + (1 - damping) * cur_estimate - hvp / scale
                for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
            ]

            # Update the Tqdm progress bar with the current norm so the user can
            # see it converge.
            if (j % 50 == 0) or (j == len(lissa_data_loader) - 1):
                norm = np.linalg.norm(_flatten_tensors(cur_estimates).cpu().numpy())
                recursion_iter.set_description(desc=f"calculating inverse HVP, norm = {norm:.5f}")

        # Accumulating X_{[i,S_2]}  (notation from the LiSSA (algo. 1) [https://arxiv.org/pdf/1602.03943.pdf]
        # Need to divide by `scale` again here because the `vs` represent gradients
        # that haven't been scaled yet.
        inverse_hvps = [
            inverse_hvp + cur_estimate / scale
            for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
        ]
    return_ihvp = _flatten_tensors(inverse_hvps)
    return_ihvp /= num_samples
    return return_ihvp


def get_hvp(
    loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """
    Get a Hessian-Vector Product (HVP) `Hv` for each Hessian `H` of the `loss`
    with respect to the one of the parameter tensors in `params` and the corresponding
    vector `v` in `vectors`.

    # Parameters

    loss : `torch.Tensor`
        The loss calculated from the output of the model.
    params : `Sequence[torch.Tensor]`
        Tunable and used parameters in the model that we will calculate the gradient and hessian
        with respect to.
    vectors : `Sequence[torch.Tensor]`
        The list of vectors for calculating the HVP.
    """
    # Sanity check before performing element-wise multiplication
    assert len(params) == len(vectors)
    assert all(p.size() == v.size() for p, v in zip(params, vectors))
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hvp = autograd.grad(grads, params, grad_outputs=vectors)
    return hvp


def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Unwraps a list of parameters gradients

    # Returns

    `torch.Tensor`
        A tensor of shape `(x,)` where `x` is the total number of entires in the gradients.
    """
    views = []
    for p in tensors:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)
