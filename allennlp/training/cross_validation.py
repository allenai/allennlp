import copy
import logging
import os
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import sklearn.model_selection
from numpy.random.mtrand import RandomState
from overrides import overrides
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import Dataset, Subset

from allennlp.commands.train import TrainModel
from allennlp.common import Lazy, Registrable, util as common_util
from allennlp.data import DataLoader, Instance, Vocabulary, DatasetReader
from allennlp.models.model import Model
from allennlp.training import Trainer, util as training_util
from allennlp.training.metrics import Average


class CrossValidator(Registrable, BaseCrossValidator):
    default_implementation = "k_fold"

    def __call__(
        self, instances: Sequence[Instance]
    ) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        groups, labels = self._labels_groups(instances)
        return super().split(instances, labels, groups=groups)

    @overrides
    def split(self, instances: Sequence[Instance]) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        return self(instances)

    @overrides
    def get_n_splits(self, instances: Sequence[Instance]) -> int:
        groups, labels = self._labels_groups(instances)
        return super().get_n_splits(instances, labels, groups=groups)

    @staticmethod
    def _labels_groups(
        instances: Sequence[Instance],
    ) -> Tuple[Optional[Sequence[Union[str, int]]], Optional[Sequence[Any]]]:
        labels = [instance["label"] for instance in instances] if "label" in instances[0] else None
        groups = (
            [instance["_group"] for instance in instances] if "_group" in instances[0] else None
        )
        return groups, labels


@CrossValidator.register("group_k_fold")
class GroupKFold(CrossValidator, sklearn.model_selection.GroupKFold):
    def __init__(self, n_splits: int = 5) -> None:
        super().__init__(n_splits=n_splits)


@CrossValidator.register("group_shuffle_split")
class GroupShuffleSplit(CrossValidator, sklearn.model_selection.GroupShuffleSplit):
    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state
        )


@CrossValidator.register("k_fold")
class KFold(CrossValidator, sklearn.model_selection.KFold):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


@CrossValidator.register("leave_one_group_out")
class LeaveOneGroupOut(CrossValidator, sklearn.model_selection.LeaveOneGroupOut):
    pass


@CrossValidator.register("leave_one_out")
class LeaveOneOut(CrossValidator, sklearn.model_selection.LeaveOneOut):
    pass


@CrossValidator.register("leave_p_groups_out")
class LeavePGroupsOut(CrossValidator, sklearn.model_selection.LeavePGroupsOut):
    def __init__(self, n_groups: int) -> None:
        super().__init__(n_groups=n_groups)


@CrossValidator.register("leave_p_out")
class LeavePOut(CrossValidator, sklearn.model_selection.LeavePOut):
    def __init__(self, p: int) -> None:
        super().__init__(p=p)


@CrossValidator.register("predefined_split")
class PredefinedSplit(CrossValidator, sklearn.model_selection.PredefinedSplit):
    def __init__(self, test_fold: Sequence[int]) -> None:
        super().__init__(test_fold=test_fold)


@CrossValidator.register("repeated_k_fold")
class RepeatedKFold(CrossValidator, sklearn.model_selection.RepeatedKFold):
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


@CrossValidator.register("repeated_stratified_k_fold")
class RepeatedStratifiedKFold(CrossValidator, sklearn.model_selection.RepeatedStratifiedKFold):
    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


@CrossValidator.register("shuffle_split")
class ShuffleSplit(CrossValidator, sklearn.model_selection.ShuffleSplit):
    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state
        )


@CrossValidator.register("stratified_k_fold")
class StratifiedKFold(CrossValidator, sklearn.model_selection.StratifiedKFold):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


@CrossValidator.register("stratified_shuffle_split")
class StratifiedShuffleSplit(CrossValidator, sklearn.model_selection.StratifiedShuffleSplit):
    def __init__(
        self,
        n_splits: int = 10,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(
            n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state
        )


@CrossValidator.register("time_series_split")
class TimeSeriesSplit(CrossValidator, sklearn.model_selection.TimeSeriesSplit):
    def __init__(self, n_splits: int = 5, max_train_size: Optional[int] = None) -> None:
        super().__init__(n_splits=n_splits, max_train_size=max_train_size)


logger = logging.getLogger(__name__)


@TrainModel.register("cross_validation", constructor="from_partial_objects")
class CrossValidateModel(TrainModel):
    def __init__(
        self,
        serialization_dir: str,
        dataset: Dataset,
        data_loader_builder: Lazy[DataLoader],
        model: Model,
        trainer_builder: Lazy[Trainer],
        cross_validator: CrossValidator,
        batch_weight_key: str = "",
        retrain: bool = False,
    ) -> None:
        data_loader = data_loader_builder.construct(dataset=dataset)

        # The "main" trainer. It's used to pass to super and when `retrain` option is set.
        #
        # We don't need to pass `serialization_dir` and `local_rank` here, because they will have
        # been passed through the trainer by from_params already, because they were keyword
        # arguments to construct this class in the first place.
        trainer = trainer_builder.construct(model=model, data_loader=data_loader)

        super().__init__(serialization_dir=serialization_dir, model=model, trainer=trainer)

        self.dataset = dataset
        self.data_loader_builder = data_loader_builder
        self.trainer_builder = trainer_builder
        self.batch_weight_key = batch_weight_key
        self.cross_validator = cross_validator
        self.retrain = retrain

    @overrides
    def run(self) -> Dict[str, Any]:
        metrics_by_fold = []

        n_splits = self.cross_validator.get_n_splits(self.dataset)

        for fold_index, (train_indices, test_indices) in enumerate(
            self.cross_validator(self.dataset)
        ):
            logger.info(f"Fold {fold_index}/{n_splits - 1}")

            serialization_dir = os.path.join(self.serialization_dir, f"fold_{fold_index}")
            os.makedirs(serialization_dir, exist_ok=True)

            train_dataset = Subset(self.dataset, train_indices)
            test_dataset = Subset(self.dataset, test_indices)

            train_data_loader = self.data_loader_builder.construct(dataset=train_dataset)
            test_data_loader = self.data_loader_builder.construct(dataset=test_dataset)

            model = copy.deepcopy(self.model)
            subtrainer = self.trainer_builder.construct(
                serialization_dir=serialization_dir, data_loader=train_data_loader, model=model
            )

            fold_metrics = subtrainer.train()

            for metric_key, metric_value in training_util.evaluate(
                model,
                test_data_loader,
                subtrainer.cuda_device,
                batch_weight_key=self.batch_weight_key,
            ).items():
                if metric_key in fold_metrics:
                    fold_metrics[f"test_{metric_key}"] = metric_value
                else:
                    fold_metrics[metric_key] = metric_value

            common_util.dump_metrics(
                os.path.join(subtrainer._serialization_dir, "metrics.json"), fold_metrics, log=True
            )

            metrics_by_fold.append(fold_metrics)

        metrics = {}

        for metric_key, fold_0_metric_value in metrics_by_fold[0].items():
            if isinstance(fold_0_metric_value, float):
                average = Average()
                for fold_index, fold_metrics in enumerate(metrics_by_fold):
                    metric_value = fold_metrics[metric_key]
                    metrics[f"fold{fold_index}_{metric_key}"] = metric_value
                    average(metric_value)
                metrics[f"average_{metric_key}"] = average.get_metric()
            else:
                for fold_index, fold_metrics in enumerate(metrics_by_fold):
                    metrics[f"fold{fold_index}_{metric_key}"] = fold_metrics[metric_key]

        if self.retrain:
            self.trainer.train()

        return metrics

    @overrides
    def finish(self, metrics: Dict[str, Any]):
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
    @overrides
    def from_partial_objects(
        cls,
        serialization_dir: str,
        batch_weight_key: str,
        dataset_reader: DatasetReader,
        data_path: str,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        cross_validator: CrossValidator,
        local_rank: int,  # It's passed transparently directly to the trainer.
        vocabulary: Lazy[Vocabulary] = None,
        retrain: bool = False,
    ) -> "CrossValidateModel":
        logger.info(f"Reading data from {data_path}")
        dataset = dataset_reader.read(data_path)

        vocabulary_ = vocabulary.construct(instances=dataset) or Vocabulary.from_instances(dataset)
        model_ = model.construct(vocab=vocabulary_)

        if common_util.is_master():
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        dataset.index_with(model_.vocab)

        return cls(
            serialization_dir=serialization_dir,
            dataset=dataset,
            model=model_,
            data_loader_builder=data_loader,
            trainer_builder=trainer,
            cross_validator=cross_validator,
            batch_weight_key=batch_weight_key,
            retrain=retrain,
        )
