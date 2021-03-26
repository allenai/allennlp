import json

from typing import List, Optional, Tuple, Union
import numpy as np
from torch import Tensor
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.nn import Parameter

from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.predictors import Predictor
from allennlp.models.model import Model
from allennlp.data import DatasetReader, Batch, Instance
from allennlp.data.data_loaders import DataLoader, MultiProcessDataLoader
from allennlp.nn import util


@InfluenceInterpreter.register("simple-influence")
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as a `SimpleInfluence` with name "simple-influence". This is a simple influence function
    calculator. We simply go through every examples in train set to calculate the influence score, and uses
    recommended LiSSA algorithm (essentially a first-order Talyor approxmation) to approximate the inverse
    of Hessian used for influence score calculation. At best, we uses a single GPU for running the calculation.

    # Parameter
    lissa_batch_size: int = 8,
        Optional. This is a hyper-parameter used by a dataloader in LiSSA algorithm.
        According to [https://arxiv.org/pdf/1703.04730.pdf], it is better to use batched samples for approximation
        for better stability.
    damping: float = 3e-3,
        Optional. This is a hyperparameter for LiSSA algorithm.
        A damping termed added in case the approximated Hessian (during LiSSA algorithm) has
        negative eigenvalues. This is a hyperparameter.
    num_samples: int = 1,
        Optional. This is a hyperparameter for LiSSA algorithm that we
        determine how many rounds of recursion process we would like to run for approxmation.
    recur_depth: Optional[Union[float, int]] = 0.25,
        Optional. This is a hyperparameter for LiSSA algorithm that we
        determine the recursion depth we would like to go through.
    scale: float = 1e4,
        Optional. This is a hyperparameter for LiSSA algorithm to tune such that the Taylor expansion converges.
    """

    def __init__(
        self,
        predictor: Predictor,
        train_data_path: str,
        train_dataset_reader: DatasetReader,
        test_dataset_reader: Optional[DatasetReader] = None,
        params_to_freeze: Optional[List[str]] = None,
        k: int = 20,
        device: int = -1,
        lissa_batch_size: int = 8,
        damping: float = 3e-3,
        num_samples: int = 1,
        recur_depth: Optional[Union[float, int]] = 0.25,
        scale: float = 1e4,
    ) -> None:
        super().__init__(
            predictor=predictor,
            train_dataset_reader=train_dataset_reader,
            test_dataset_reader=test_dataset_reader,
            train_data_path=train_data_path,
            params_to_freeze=params_to_freeze,
            k=k,
            device=device,
        )
        self._lissa_batch_size = lissa_batch_size
        self._lissa_dataloader = MultiProcessDataLoader(
            self.train_dataset_reader, train_data_path, batch_size=self._lissa_batch_size
        )
        self._lissa_dataloader.set_target_device(self._device)
        self._lissa_dataloader.index_with(self.vocab)

        self.num_samples = num_samples
        self.damping = damping
        self.recur_depth = recur_depth
        self.scale = scale

    def interpret_and_save(self, test_data_path: str, output_file: str):
        """
        This is the "main" function of influence score calcualtion. This function will go through
        example by example in the provided test set, and run the LiSSA algorithm to
        approximate the inverse Hessian for each test examples. Then, it will use this inverse
        to calculate the score for each train examples. As a result, we will output
        `k` examples with the highest and `k` examples with the lowest influence scores.
        The output file contains lines of dictionary (one per line). Each line looks like:
            {
                "test_instance": {<input and output field in human readble form>, "loss": ...}
                "top_{k}_train_instances": [{<same format as test_instance>} {<....>}],
                "bottom_{k}_train_instances": [{<same format as test_instance>} {<....>}]
            }

        # Parameter
        test_data_path: `str`
            Required. This is the file path to the test data. Here, we make the assumption that
            each instance in the test file will contain as least information as each one from train file.

        output_file: `str`
            Required. This is the path way to save output. Here we assume the directory contained
            in the path is valid

        """
        test_dataloader = MultiProcessDataLoader(
            self.test_dataset_reader, test_data_path, batch_size=1
        )
        test_dataloader.set_target_device(self._device)
        test_dataloader.index_with(self.vocab)
        output_content = []
        for test_idx, test_instance in enumerate(
            tqdm(test_dataloader._instances, desc="Test set index")
        ):
            test_instance: Instance

            test_instance_dict = test_instance.human_readable_dict()
            test_batch = Batch([test_instance])
            test_batch.index_instances(self.vocab)
            self.model.eval()
            dataset_tensor_dict = util.move_to_device(test_batch.as_tensor_dict(), self._device)
            test_output_dict = self.model(**dataset_tensor_dict)

            # get test example's gradient
            test_loss = test_output_dict["loss"]
            test_instance_dict["loss"] = test_loss.detach().item()
            # We record information about the test instance
            output_per_test = {"test_instance": test_instance_dict}

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
            recursion_depth = self.recur_depth
            if type(recursion_depth) is float:
                recursion_depth = int(
                    len(self._lissa_dataloader)
                    * self._lissa_dataloader.batch_size
                    * recursion_depth
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
                train_loss = train_output_dict["loss"]
                train_outputs.append({"loss": train_loss.detach().item()})

                self.model.zero_grad()
                train_grads = autograd.grad(train_loss, self._used_params)
                # dL_test * d theta as in 2.2 of [https://arxiv.org/pdf/2005.06676.pdf]
                influences[train_idx] = torch.dot(inv_hvp, self.flatten_tensors(train_grads)).item()

            # For each test example, we output top-k and training instances
            # Then, we record the top-k training instances
            _, indices = torch.topk(torch.tensor(influences), self._k)
            top_k_train_instances = []
            for idx in indices:
                train_instance = self._train_loader._instances[idx]
                train_instance_dict = train_instance.human_readable_dict()
                train_instance_dict["loss"] = train_outputs[idx]["loss"]
                top_k_train_instances.append(train_instance_dict)
            output_per_test[f"top_{self._k}_train_instances"] = top_k_train_instances

            # Then, we record the bottom-k training instances
            _, indices = torch.topk(-torch.tensor(influences), self._k)
            assert len(train_outputs) == len(influences)
            bottom_k_train_instances = []
            for idx in indices:
                train_instance = self._train_loader._instances[idx]
                train_instance_dict = train_instance.human_readable_dict()
                train_instance_dict["loss"] = train_outputs[idx]["loss"]
                bottom_k_train_instances.append(train_instance_dict)
            output_per_test[f"bottom_{self._k}_train_instances"] = bottom_k_train_instances

            output_content.append(output_per_test)

        with open(output_file, "w") as f:
            for line in output_content:
                json.dump(line, f)
                f.write("\n")

    @staticmethod
    def get_inverse_hvp_lissa(
        vs: List[torch.Tensor],
        model: Model,
        used_params: List[Union[Parameter, Tensor]],
        lissa_dataloader: DataLoader,
        num_samples: int,
        recursion_depth: int,
        damping: float,
        scale: float,
    ) -> torch.Tensor:
        """
        This function approximates the inverse of Hessian-Vector Product(HVP) w.r.t. the input

        # Parameters

        vs :   `List[torch.Tensor]`
            `v` as in "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
            Due to how model return its gradient, `v` is returned as a list of gradients.

        model :  `Model`
            the model we will take gradient/Hessian w.r.t.
        used_params :   `List[Parameter]`
            a list of parameters that's used by the model
        lissa_data_loader :   `DataLoader`
            a dataloader that sample data points for LiSSA algorithm
        num_samples :   `int`
            number of sample we will use to approximate the inverse Hessian by recursion
        recursion_depth :   `int`
            number of recursion to run in approximating the inverse
        damping :   `float`


        """
        inverse_hvps = [0 for _ in vs]
        for _ in range(num_samples):  # i.e. number of recursion
            # See a explanation at "Stochastic estimation" paragraph in [https://arxiv.org/pdf/1703.04730.pdf]
            # initialize \tilde{H}^{−1}_0 v = v
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
                # Hessian of loss @ \tilde{H}^{−1}_{j - 1} v
                hvps = SimpleInfluence.get_hessian_vector_product(
                    train_output_dict["loss"], used_params, cur_estimates
                )

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
                        f"{np.linalg.norm(SimpleInfluence.flatten_tensors(cur_estimates).cpu().numpy())}"
                    )

            # accumulating X_{[i,S_2]}  (notation from the LiSSA (algo. 1) [https://arxiv.org/pdf/1602.03943.pdf]
            inverse_hvps = [
                inverse_hvp + cur_estimate / scale
                for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
            ]
        return_ihvp = SimpleInfluence.flatten_tensors(inverse_hvps)
        return_ihvp /= num_samples
        return return_ihvp

    @staticmethod
    def flatten_tensors(tensors: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
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
    ) -> Tuple[torch.Tensor]:
        """
        Get a Hessian-Vector Product (HVP) from the loss of a model. This is equivalent to:
            1. Calculate parameters gradient;
            2. Element-wise multiply each gradient with `vector` of the same size
            3. For each product obtained (e.g. a matrix), calculate the gradient w.r.t. parameters.

        Note here that we are taking gradient (Step 3) of gradient (Step 1), so we have Hessian as our results.
        See [https://github.com/kohpangwei/influence-release/blob/master/influence/hessians.py]

        Why we need `vector`? It turns out HVP is a common operations in many optimization operations.
        # Parameters

        loss :   `torch.Tensor`
            loss caluclated from the output of a model
        model_params :   `List[Parameter]`
            tunable and used parameters in the model that
            we will calculate gradient/hessian with respect to
        vectors :    `List[torch.Tensor]`
            List of "vectors" for calculating the Hessian-"Vector" Product.
            Note a vector can be a unwrapped version of a matrix
        """
        # Sanity check before performing element-wise multiplication
        assert len(params) == len(vectors)
        assert all(p.size() == v.size() for p, v in zip(params, vectors))
        grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
        hvp = autograd.grad(grads, params, grad_outputs=vectors)
        return hvp
