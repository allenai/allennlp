import math

from typing import List, Optional
import numpy as np
import torch
import torch.autograd as autograd
from torch.nn import Parameter

from allennlp.common.util import JsonDict, sanitize, get_frozen_and_tunable_parameter_names
from allennlp.interpret.influence_interpreters.influence_interpreter import (
    InfluenceInterpreter,
)
from allennlp.predictors import Predictor
from allennlp.models.model import Model
from allennlp.data import DatasetReader, PyTorchDataLoader
from allennlp.data.batch import Batch
from allennlp.nn import util


@InfluenceInterpreter.register("simple-influence")
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as a `SimpleInfluence` with name "simple-influence".
    """

    def __init__(self,
                 predictor: Predictor,
                 train_filepath: str,
                 test_filepath: str,
                 train_dataset_reader: DatasetReader,
                 test_dataset_reader: DatasetReader = None,
                 train_batch_size: int = 8,
                 k: int = 20,
                 device: int = -1,
                 damping: float = 3e-3,
                 num_recur: int = 1,
                 recur_depth: float = 0.25,
                 scale: float = 1e4,) -> None:
        super().__init__(predictor, train_dataset_reader, test_dataset_reader,
                         train_filepath, test_filepath, train_batch_size, k, device)
        self.num_recur = num_recur
        self.damping = damping
        self.recur_depth = recur_depth
        self.scale = scale

    def influence_interpret_from_json(self, inputs: JsonDict) -> JsonDict:
        """
        Interpret the model's prediction on input with respect to the training instances.
        """
        labeled_instances = self.predictor.json_to_labeled_instances(inputs)

        v = test_grads

        # inverse Hessian-Vector Product
        ihvp = None
        for _ in range(self.num_recur):
            cur_estimate = v

    def calculate_inflence_and_save(self, output_file):
        for instance in self._test_set:
            self.model.eval()
            # prediction = self.predictor.predict_instance(instance)
            input_as_tensor_dict = Batch([instance]).as_tensor_dict()
            raw_prediction = self.model(**input_as_tensor_dict)

            # get test example's gradient
            test_loss = raw_prediction["loss"]
            self.model.zero_grad()

            if self._used_params is None:
                test_loss.backward(retain_graph=True)
                self._used_params_name = [n for n, p in self.model.named_parameters()
                                     if p.requires_grad and p.grad is not None]
                self._used_params = [p for _, p in self.model.named_parameters()
                                     if p.requires_grad and p.grad is not None]
                test_grads = [p.grad for _, p in self.model.named_parameters()
                                     if p.requires_grad and p.grad is not None]
            else:
                test_grads = autograd.grad(test_loss, self._used_params)

            assert len(test_grads) == len(self._used_params)
            # assert all(test_grads[i][0] == self._params[i][0] for i in range(len(test_grads)))

            inv_hvp = self._get_inverse_hvp_lissa(test_grads)
            print()
            influences = np.zeros(len(self._train_set))

            train_tok_sal_lists = []
            for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids, _) in enumerate(
                    tqdm(train_dataloader, desc="Train set index")):
                model.train()
                _input_ids = _input_ids.to(device)
                _input_mask = _input_mask.to(device)
                _segment_ids = _segment_ids.to(device)
                _label_ids = _label_ids.to(device)

                ######## L_TRAIN GRADIENT ########
                model.zero_grad()
                train_loss = model(_input_ids, _segment_ids, _input_mask, _label_ids)
                train_grads = autograd.grad(train_loss, param_influence)
                influences[train_idx] = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()
                ################

            pass

    def _get_inverse_hvp_lissa(self, v):
        """
        This function approximates the inverse of Hessian-Vector Product(HVP) w.r.t. the input
        :param test_grads:
        :return:
        """
        recursion_depth = int(len(self._train_set) * self.recur_depth)
        # TODO: finish here
        ihvp = None
        for _ in range(self.num_recur):
            cur_estimate = v
            lissa_data_iterator = iter(self._train_loader)
            for j in range(recursion_depth):
                try:
                    training_batch = next(lissa_data_iterator)
                except StopIteration:
                    lissa_data_iterator = iter(self._train_loader)
                    training_batch = next(lissa_data_iterator)
                self.model.zero_grad()

                train_loss = self.model(**training_batch.as_tensor_dict())
                hvp = self.get_hessian_vector(train_loss, self._used_params, cur_estimate)
                cur_estimate = [_a + (1 - self.damping) * _b - _c / self.scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
                if (j % 200 == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %f" % (
                    j, np.linalg.norm(self.gather_flat_grad(cur_estimate).cpu().numpy())))
            if ihvp is None:
                ihvp = [_a / self.scale for _a in cur_estimate]
            else:
                ihvp = [_a + _b / self.scale for _a, _b in zip(ihvp, cur_estimate)]
        return_ihvp = self.gather_flat_grad(ihvp)
        return_ihvp /= self.num_samples
        return return_ihvp

    @staticmethod
    def gather_flat_grad(grads: List[Parameter]):
        views = []
        for p in grads:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @staticmethod
    def get_hessian_vector(loss, model_params, v):
        grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
        hv = autograd.grad(grad, model_params, grad_outputs=v)
        return hv

