import math

from typing import List, Optional, Tuple
import numpy as np
from torch import Tensor
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.nn import Parameter

from allennlp.common.util import JsonDict, sanitize, get_frozen_and_tunable_parameter_names
from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.predictors import Predictor
from allennlp.models.model import Model
from allennlp.data import DatasetReader
from allennlp.data.data_loaders import DataLoader, MultiProcessDataLoader
from allennlp.data.batch import Batch
from allennlp.nn import util


@InfluenceInterpreter.register("simple-influence")
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as a `SimpleInfluence` with name "simple-influence".
    """

    def __init__(
        self,
        predictor: Predictor,
        train_filepath: str,
        test_filepath: str,
        train_dataset_reader: DatasetReader,
        test_dataset_reader: DatasetReader = None,
        train_batch_size: int = 8,
        k: int = 20,
        device: int = -1,
        damping: float = 3e-3,
        num_samples: int = 1,
        recur_depth: float = 0.25,
        scale: float = 1e4,
    ) -> None:
        super().__init__(
            predictor,
            train_dataset_reader,
            test_dataset_reader,
            train_filepath,
            test_filepath,
            train_batch_size,
            k,
            device,
        )
        self._train_batch_size = train_batch_size
        self._lissa_dataloader = MultiProcessDataLoader(
            self.train_dataset_reader, train_filepath, batch_size=self._train_batch_size
        )
        self._lissa_dataloader.set_target_device(self._device)
        self._lissa_dataloader.index_with(self.vocab)

        # TODO: make the following loader to be deterministic
        # think how to reuse train_set
        self._train_loader = MultiProcessDataLoader(
            self.train_dataset_reader, train_filepath, batch_size=1
        )
        self._train_loader.set_target_device(self._device)
        self._train_loader.index_with(self.vocab)

        self._test_loader = MultiProcessDataLoader(
            self.test_dataset_reader, test_filepath, batch_size=1
        )
        self._test_loader.set_target_device(self._device)
        self._test_loader.index_with(self.vocab)

        self._k = min(self._k, len(self._train_loader))
        self.num_samples = num_samples
        self.damping = damping
        self.recur_depth = recur_depth
        self.scale = scale

    def calculate_inflence_and_save(self, output_file):
        output_content = []
        for test_idx, test_batch in enumerate(tqdm(self._test_loader, desc="Test set index")):
            self.model.eval()
            test_output_dict = self.model(**test_batch)

            # get test example's gradient
            test_loss = test_output_dict["loss"]
            self.model.zero_grad()

            if self._used_params is None:
                # we only know what parameters in the models requires gradient after
                # we do the first .backward() and we store those used parameters
                test_loss.backward(retain_graph=True)
                self._used_params = [
                    p
                    for _, p in self.model.named_parameters()
                    if p.requires_grad and p.grad is not None
                ]

            # list of (parameters) gradients w.r.t. the test loss
            test_grads: List[Tensor] = autograd.grad(test_loss, self._used_params)
            assert len(test_grads) == len(self._used_params)

            # Approximate the inverse of Hessian-Vector Product through LiSSA algorithm
            recursion_depth = int(
                len(self._lissa_dataloader) * self._lissa_dataloader.batch_size * self.recur_depth
            )
            inv_hvp = self.get_inverse_hvp_lissa(
                vs=test_grads,
                model=self.model,
                used_params=self._used_params,
                lissa_dataloader=self._lissa_dataloader,
                num_samples=self.num_samples,
                recursion_depth=recursion_depth,
                damping=self.damping,
                scale=self.scale,
            )

            # Now, w.r.t. the current test examples, we iterate through the train set again to gain
            # calculate the influence score for each training examples.
            influences = torch.zeros(len(self._train_loader))
            train_outputs = []
            for train_idx, training_batch in enumerate(
                tqdm(self._train_loader, desc="Train set index")
            ):
                self.model.train()
                train_output_dict = self.model(**training_batch)
                train_outputs.append({"loss": train_output_dict["loss"].item()})

                self.model.zero_grad()
                train_grads = autograd.grad(train_output_dict["loss"], self._used_params)
                # dL_test * d theta as in 2.2 of [https://arxiv.org/pdf/2005.06676.pdf]
                influences[train_idx] = torch.dot(inv_hvp, self.flatten_tensors(train_grads)).item()

            # For each test example, we only output top-k training instances
            _, indices = torch.topk(torch.tensor(influences), self._k)
            # We record information about the test instance
            per_test_output = {
                "test_instance": {
                    "tokens": " ".join(str(t) for t in test_batch["tokens"].tokens[0]),
                    "label": test_batch["label"].label,
                    "loss": test_output_dict["loss"].item(),
                }
            }

            # Then, we record the top-k training instances
            assert len(train_outputs) == len(influences)
            top_k_train_instances = []
            for idx in indices:
                train_instance = self._train_loader._instances[idx]
                train_output = train_outputs[idx]
                top_k_train_instances.append(
                    {
                        "tokens": " ".join(str(t) for t in train_instance["tokens"].tokens),
                        "label": train_instance["label"].label,
                        "loss": train_output["loss"],
                    }
                )
            per_test_output[f"top_{self._k}_train_instances"] = top_k_train_instances
            output_content.append(per_test_output)

        print()
        pass

    @staticmethod
    def get_inverse_hvp_lissa(
        vs: List[torch.Tensor],
        model: Model,
        used_params: List[Parameter],
        lissa_dataloader: DataLoader,
        num_samples: int,
        recursion_depth: int,
        damping: float,
        scale: float,
    ) -> torch.Tensor:
        """
        This function approximates the inverse of Hessian-Vector Product(HVP) w.r.t. the input

        # Parameters

        vs: `v` as in "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
            Due to how model return its gradient, `v` is returned as a list of gradients.

        model: the model we will take gradient/Hessian w.r.t.
        used_params: a list of parameters that's used by the model
        lissa_data_loader: a dataloader that sample data points

        """
        inverse_hvps = [0 for _ in vs]
        for _ in range(num_samples):  # i.e. number of recursion
            cur_estimates = vs
            lissa_data_iterator = iter(lissa_dataloader)
            for j in range(recursion_depth):
                try:
                    training_batch = next(lissa_data_iterator)
                except StopIteration:
                    # re-initialize a data loader to continue the recursion
                    lissa_data_iterator = iter(lissa_dataloader)
                    training_batch = next(lissa_data_iterator)
                model.zero_grad()
                train_output_dict = model(**training_batch)

                # sample a batch and calculate the gradient
                hvps = SimpleInfluence.get_hessian_vector_product(
                    train_output_dict["loss"], used_params, cur_estimates
                )

                # this is the recursive step in the LiSSA (algo. 1) [https://arxiv.org/pdf/1602.03943.pdf]
                # See a explanation at "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
                # v + (I - Hessian_at_x) * cur_estimate = v + cur_estimate - Hessian_at_x * cur_estimate
                cur_estimates = [
                    v + (1 - damping) * cur_estimate - hvp / scale
                    for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
                ]
                # Manually checking if it converges
                if (j % 200 == 0) or (j == recursion_depth - 1):
                    print(
                        f"Recursion at depth {j}: norm is "
                        f"{np.linalg.norm(SimpleInfluence.flatten_tensors(cur_estimates).cpu().numpy())}"
                    )

            # accumulating X_{[i,S_2]}
            inverse_hvps = [
                inverse_hvp + cur_estimate / scale
                for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
            ]
        return_ihvp = SimpleInfluence.flatten_tensors(inverse_hvps)
        return_ihvp /= num_samples
        return return_ihvp

    @staticmethod
    def flatten_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
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

    @staticmethod
    def get_hessian_vector_product(
        loss: torch.Tensor, params: List[Parameter], vectors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Get a Hessian-Vector Product (HVP) from the loss of a model. This is equivalent to:
            1. Calculate parameters gradient;
            2. Element-wise multiply each gradient with `vector` of the same size
            3. For each product obtained (e.g. a matrix), calculate the gradient w.r.t. parameters.

        Note here that we are taking gradient (Step 3) of gradient (Step 1), so we have Hessian as our results.
        See [https://github.com/kohpangwei/influence-release/blob/master/influence/hessians.py]

        Why we need `vector`? It turns out HVP is a common operations in many optimization operations.
        # Parameters

        loss: `torch.Tensor`
            loss caluclated from the output of a model
        model_params: `List[Parameter]`
            tunable and used parameters in the model that
            we will calculate gradient/hessian with respect to
        vectors: `List[torch.Tensor]`
            List of "vectors" for calculating the Hessian-"Vector" Product.
            Note a vector can be a unwrapped version of a matrix
        """
        # Sanity check before performing element-wise multiplication
        assert len(params) == len(vectors)
        assert all(p.size() == v.size() for p, v in zip(params, vectors))
        grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
        hvp = autograd.grad(grads, params, grad_outputs=vectors)
        return hvp
