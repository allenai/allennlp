from typing import List, Optional, Tuple, Union, Sequence

import numpy as np
from overrides import overrides
import torch
import torch.autograd as autograd

from allennlp.common import Lazy
from allennlp.data import DatasetReader, DatasetReaderInput, Instance
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader, MultiProcessDataLoader
from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.models.model import Model


@InfluenceInterpreter.register("simple-influence")
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as an `InfluenceInterpreter` with name "simple-influence".

    This goes through every examples in train set to calculate the influence score, and uses
    the LiSSA algorithm (essentially a first-order Talyor approxmation) to approximate the inverse
    of the Hessian used for influence score calculation.

    # Parameters

    lissa_data_loader : `Lazy[DataLoader]`, optional (default = `Lazy(MultiProcessDataLoader)`)
        The data loader used in LiSSA algorithm.
        According to [https://arxiv.org/pdf/1703.04730.pdf](https://arxiv.org/pdf/1703.04730.pdf),
        it is better to use batched samples for approximation for better stability.

    damping : `float`, optional (default = `3e-3`)
        This is a hyperparameter for LiSSA algorithm.
        A damping termed added in case the approximated Hessian (during LiSSA algorithm) has
        negative eigenvalues.

    num_samples : `int`, optional (default = `1`)
        Optional. This is a hyperparameter for LiSSA algorithm that we
        determine how many rounds of recursion process we would like to run for approxmation.

    recursion_depth : `Union[float, int]`, optional (default = `0.25`)
        This is a hyperparameter for LiSSA algorithm that
        determines the recursion depth we would like to go through.
        If a `float`, it means X% of the training examples.
        If an `int`, it means recurse for X times.

    scale : `float`, optional, (default = `1e4`)
        This is a hyperparameter for LiSSA algorithm to tune such that the Taylor expansion converges.
    """

    def __init__(
        self,
        model: Model,
        train_data_path: DatasetReaderInput,
        train_dataset_reader: DatasetReader,
        test_dataset_reader: Optional[DatasetReader] = None,
        train_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader),
        test_data_loader: Lazy[DataLoader] = Lazy(SimpleDataLoader),
        params_to_freeze: List[str] = None,
        device: int = -1,
        lissa_data_loader: Lazy[DataLoader] = Lazy(
            MultiProcessDataLoader, contructor_extras={"batch_size": 8}
        ),
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
            device=device,
        )

        self._lissa_dataloader = lissa_data_loader.construct(
            reader=train_dataset_reader, data_path=train_data_path
        )
        self._lissa_dataloader.set_target_device(self._device)
        self._lissa_dataloader.index_with(self._vocab)

        self._damping = damping
        self._num_samples = num_samples
        self._recursion_depth = recursion_depth
        self._scale = scale

    @overrides
    def calculate_influence_scores(
        self, test_instance: Instance, test_loss: float, test_grads: Sequence[torch.Tensor]
    ) -> List[float]:
        # Approximate the inverse of Hessian-Vector Product through LiSSA algorithm
        if isinstance(self._recursion_depth, float):
            recursion_depth = int(
                len(list(self._lissa_dataloader.iter_instances())) * self._recursion_depth
            )
        elif isinstance(self._recursion_depth, int):
            recursion_depth = self._recursion_depth
        else:
            raise ValueError("'recursion_depth' shoudl be a float or an int")

        inv_hvp = get_inverse_hvp_lissa(
            test_grads,
            self._model,
            self._used_params,
            self._lissa_dataloader,
            self._damping,
            self._num_samples,
            recursion_depth,
            self._scale,
        )
        return [
            # dL_test * d theta as in 2.2 of [https://arxiv.org/pdf/2005.06676.pdf]
            torch.dot(inv_hvp, flatten_tensors(x.grads)).item()
            for x in self.train_instances
        ]


def get_inverse_hvp_lissa(
    vs: Sequence[torch.Tensor],
    model: Model,
    used_params: Sequence[torch.Tensor],
    lissa_data_loader: DataLoader,
    damping: float,
    num_samples: int,
    recursion_depth: int,
    scale: float,
) -> torch.Tensor:
    """
    This function approximates the inverse of Hessian-Vector Product(HVP) w.r.t. the input.
    """
    inverse_hvps = [torch.tensor(0) for _ in vs]
    for _ in range(num_samples):  # i.e. number of recursion
        # See a explanation at "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
        # initialize \tilde{H}^{−1}_0 v = v
        cur_estimates = vs
        lissa_data_iterator = iter(lissa_data_loader)
        for j in range(recursion_depth):
            try:
                training_batch = next(lissa_data_iterator)
            except StopIteration:
                # re-initialize a data loader to continue the recursion
                lissa_data_iterator = iter(lissa_data_loader)
                training_batch = next(lissa_data_iterator)
            model.zero_grad()
            train_output_dict = model(**training_batch)

            # sample a batch and calculate the gradient
            # Hessian of loss @ \tilde{H}^{−1}_{j - 1} v
            hvps = get_hessian_vector_product(train_output_dict["loss"], used_params, cur_estimates)

            # this is the recursive step
            # cur_estimate = \tilde{H}^{−1}_{j - 1} v
            # (i.e. Hessian-Vector Product estimate from last iteration)
            # v + (I - Hessian_at_x) * cur_estimate = v + cur_estimate - Hessian_at_x * cur_estimate
            # Updating for \tilde{H}^{−1}_j v
            cur_estimates = [
                v + (1 - damping) * cur_estimate - hvp / scale
                for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
            ]
            # Manually checking if it converges
            if (j % 200 == 0) or (j == recursion_depth - 1):
                # I manually check, the norm does converge (though slowly)
                print(
                    f"Recursion at depth {j}: norm is "
                    f"{np.linalg.norm(flatten_tensors(cur_estimates).cpu().numpy())}"
                )

        # accumulating X_{[i,S_2]}  (notation from the LiSSA (algo. 1) [https://arxiv.org/pdf/1602.03943.pdf]
        inverse_hvps = [
            inverse_hvp + cur_estimate / scale
            for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
        ]
    return_ihvp = flatten_tensors(inverse_hvps)
    return_ihvp /= num_samples
    return return_ihvp


def flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Un-wraps a list of parameters gradients

    # Returns

    `torch.Tensor`
        a tensor of shape (x,) where x is the total number of entires in the gradients.
    """
    views = []
    for p in tensors:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def get_hessian_vector_product(
    loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """
    Get a Hessian-Vector Product (HVP) from the loss of a model. This is equivalent to:
        1. Calculate parameters gradient;
        2. Element-wise multiply each gradient with `vector` of the same size
        3. For each product obtained (e.g. a matrix), calculate the gradient w.r.t. parameters.

    Note here that we are taking gradient (Step 3) of gradient (Step 1), so we have Hessian as our results.
    See [github.com/kohpangwei/influence-release]
    (https://github.com/kohpangwei/influence-release/blob/master/influence/hessians.py).

    Why we need `vector`? It turns out HVP is a common operations in many optimization operations.

    # Parameters

    loss : `torch.Tensor`
        loss caluclated from the output of a model
    params : `Sequence[torch.Tensor]`
        Tunable and used parameters in the model that we will calculate the gradient/hession
        with respect to.
    vectors : `Sequence[torch.Tensor]`
        List of "vectors" for calculating the Hessian-"Vector" Product.
        Note a vector can be a unwrapped version of a matrix
    """
    # Sanity check before performing element-wise multiplication
    assert len(params) == len(vectors)
    assert all(p.size() == v.size() for p, v in zip(params, vectors))
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hvp = autograd.grad(grads, params, grad_outputs=vectors)
    return hvp
