import copy
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from overrides import overrides
from sklearn.model_selection import BaseCrossValidator, GroupKFold, KFold, LeaveOneGroupOut, LeaveOneOut, \
    LeavePGroupsOut, LeavePOut, StratifiedKFold, TimeSeriesSplit
import torch

from allennlp.common import Params, Registrable
from allennlp.common.util import dump_metrics
from allennlp.data import DataIterator, Instance
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.metrics import Average
from allennlp.training.no_op_trainer import NoOpTrainer
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CrossValidationSplitter(Registrable):
    def __init__(self, cross_validator: BaseCrossValidator) -> None:
        super().__init__()
        self.cross_validator = cross_validator

    def __call__(self, instances: List[Instance], groups: Optional[Any] = None) -> Iterable[Tuple[np.ndarray,
                                                                                                  np.ndarray]]:
        return self.cross_validator.split(instances, groups=groups)

    def get_n_splits(self, instances: List[Instance], groups: Optional[Any] = None) -> int:
        return self.cross_validator.get_n_splits(instances, groups=groups)


class _CrossValidationSplitterWrapper:
    def __init__(self, cross_validator_class: Type[BaseCrossValidator]) -> None:
        super().__init__()
        self.cross_validator_class = cross_validator_class

    def from_params(self, params: Params) -> CrossValidationSplitter:
        cross_validator = self.cross_validator_class(**params.as_dict())
        return CrossValidationSplitter(cross_validator)


CrossValidationSplitter.register('group_k_fold')(_CrossValidationSplitterWrapper(GroupKFold))
CrossValidationSplitter.register('k_fold')(_CrossValidationSplitterWrapper(KFold))
CrossValidationSplitter.register('leave_one_group_out')(_CrossValidationSplitterWrapper(LeaveOneGroupOut))
CrossValidationSplitter.register('leave_one_out')(_CrossValidationSplitterWrapper(LeaveOneOut))
CrossValidationSplitter.register('leave_p_groups_out')(_CrossValidationSplitterWrapper(LeavePGroupsOut))
CrossValidationSplitter.register('leave_p_out')(_CrossValidationSplitterWrapper(LeavePOut))
CrossValidationSplitter.register('stratified_k_fold')(_CrossValidationSplitterWrapper(StratifiedKFold))
CrossValidationSplitter.register('time_series_split')(_CrossValidationSplitterWrapper(TimeSeriesSplit))


@TrainerBase.register('cross_validation')
class CrossValidationTrainer(TrainerBase):
    SUPPORTED_SUBTRAINER_TYPES = ['default', 'no_op']

    def __init__(self,
                 model: Model,
                 train_dataset: List[Instance],
                 iterator: DataIterator,
                 subtrainer_params: Params,
                 cross_validation_splitter: CrossValidationSplitter,
                 serialization_dir: str,
                 validation_dataset: Optional[List[Instance]] = None,
                 group_key: Optional[str] = None,
                 leave_model_trained: bool = False,
                 recover: bool = False) -> None:  # FIXME: does recover make sense? Maybe to continue the CV.
        super().__init__(serialization_dir, cuda_device=-1)
        self.model = model
        self.train_dataset = train_dataset
        self.iterator = iterator
        self.subtrainer_params = subtrainer_params
        self.cross_validation_splitter = cross_validation_splitter
        self.group_key = group_key
        self.leave_model_trained = leave_model_trained
        self.validation_dataset = validation_dataset
        self.recover = recover

    def _build_subtrainer(self, serialization_dir: str, train_dataset: Iterable[Instance],
                          validation_dataset: Optional[Iterable[Instance]] = None) -> TrainerBase:
        params = self.subtrainer_params.duplicate()

        subtrainer_type = params.pop('type', 'default')

        if subtrainer_type == 'default':
            return Trainer.from_params(model=copy.deepcopy(self.model),
                                       serialization_dir=serialization_dir,
                                       iterator=self.iterator,
                                       train_data=train_dataset,
                                       validation_data=validation_dataset,
                                       params=params)
        elif subtrainer_type == 'no_op':
            return NoOpTrainer(serialization_dir, self.model)
        else:
            raise ValueError(f"Subtrainer type '{subtrainer_type}' not supported."
                             f" Supported types are: {self.SUPPORTED_SUBTRAINER_TYPES}")

    def _get_groups(self, dataset: Iterable[Instance]) -> Optional[List[torch.Tensor]]:
        if self.group_key:
            for instance in dataset:
                instance.index_fields(self.iterator.vocab)

            return [instance[self.group_key].as_tensor(instance.get_padding_lengths()[self.group_key])
                    for instance in dataset]
        else:
            return None

    @overrides
    def train(self) -> Dict[str, float]:
        metrics_by_fold = []

        if self.validation_dataset:
            logger.info("Using the concatenation of the training and the validation datasets for"
                        " cross-validation.")
            dataset = self.train_dataset + self.validation_dataset
        else:
            dataset = self.train_dataset

        groups = self._get_groups(dataset)

        n_splits = self.cross_validation_splitter.get_n_splits(dataset, groups=groups)

        for fold_index, (train_indices, validation_indices) in enumerate(
                self.cross_validation_splitter(dataset, groups=groups)):
            logger.info("Fold %d/%d", fold_index, n_splits - 1)
            serialization_dir = os.path.join(self._serialization_dir, f'fold_{fold_index}')
            os.makedirs(serialization_dir, exist_ok=True)

            train_dataset = [dataset[i] for i in train_indices]
            validation_dataset = [dataset[i] for i in validation_indices]

            subtrainer = self._build_subtrainer(serialization_dir, train_dataset, validation_dataset)

            # try:
            fold_metrics = subtrainer.train()
            # except KeyboardInterrupt:  # TODO
            #     # if we have completed an epoch, try to create a model archive.
            #     if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            #         logging.info("Training interrupted by the user. Attempting to create "
            #                      "a model archive using the current best epoch weights.")
            #         archive_model(serialization_dir)
            #     raise

            # archive_model(serialization_dir)  # TODO

            if not fold_metrics:
                # pylint: disable=protected-access
                fold_metrics = training_util.evaluate(self.model, validation_dataset, self.iterator,
                                                      cuda_device=subtrainer._cuda_devices[0], batch_weight_key='')

            dump_metrics(os.path.join(serialization_dir, 'metrics.json'), fold_metrics, log=True)

            metrics_by_fold.append(fold_metrics)

        metrics = {}

        for metric_key, fold_0_metric_value in metrics_by_fold[0].items():
            if isinstance(fold_0_metric_value, float):
                average = Average()
                for fold_index, fold_metrics in enumerate(metrics_by_fold):
                    metric_value = fold_metrics[metric_key]
                    metrics[f'fold{fold_index}_{metric_key}'] = metric_value
                    average(metric_value)
                metrics[f'average_{metric_key}'] = average.get_metric()

        if self.leave_model_trained:
            subtrainer = self._build_subtrainer(self._serialization_dir, self.train_dataset,
                                                self.validation_dataset)
            subtrainer.train()

        return metrics

    @classmethod
    @overrides
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False) -> 'CrossValidationTrainer':
        # pylint: disable=arguments-differ
        trainer_pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        if not isinstance(trainer_pieces.train_dataset, list):
            raise ValueError("The training dataset must be a list. DatasetReader's lazy mode is not supported.")
        if trainer_pieces.validation_dataset and not isinstance(trainer_pieces.validation_dataset, list):
            raise ValueError("The validation dataset must be a list. DatasetReader's lazy mode is not supported.")

        trainer_params = trainer_pieces.params
        subtrainer_params = trainer_params.pop('trainer')
        cross_validation_splitter = CrossValidationSplitter.from_params(trainer_params.pop('splitter'))
        group_key = trainer_params.pop('group_key', None)
        # If there's a test dataset then probably we want to leave the model trained with all the data at the end.
        leave_model_trained = trainer_params.pop('leave_model_trained', bool(trainer_pieces.test_dataset))
        trainer_params.assert_empty(__name__)

        params.assert_empty(__name__)
        return cls(model=trainer_pieces.model,  # type: ignore
                   train_dataset=trainer_pieces.train_dataset,
                   iterator=trainer_pieces.iterator,
                   subtrainer_params=subtrainer_params,
                   cross_validation_splitter=cross_validation_splitter,
                   serialization_dir=serialization_dir,
                   validation_dataset=trainer_pieces.validation_dataset,
                   group_key=group_key,
                   leave_model_trained=leave_model_trained,
                   recover=recover)
