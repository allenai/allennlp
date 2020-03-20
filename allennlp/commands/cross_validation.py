import copy
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

from torch.utils.data import Dataset, Subset

from allennlp.commands.cross_validator import CrossValidator, default_get_groups, default_get_labels
from allennlp.commands.train import TrainModel
from allennlp.common import Lazy, Registrable, util as common_util
from allennlp.data import DataLoader, DatasetReader, Vocabulary, Instance
from allennlp.models.model import Model
from allennlp.training import Trainer, util as training_util
from allennlp.training.metrics import Average

logger = logging.getLogger(__name__)


def instances_get_key(instances: Sequence[Instance], key: str) -> Sequence[Any]:
    return [instance[key] for instance in instances]


@TrainModel.register("cross_validation", constructor="from_partial_objects")
class CrossValidateModel(Registrable):
    def __init__(
        self,
        serialization_dir: str,
        dataset: Dataset,
        data_loader_builder: Lazy[DataLoader],
        model: Model,
        trainer_builder: Lazy[Trainer],
        cross_validator: CrossValidator,
        instances_labels_fn: Callable[
            [Sequence[Instance]], Optional[Sequence[Any]]
        ] = default_get_labels,
        instances_groups_fn: Callable[
            [Sequence[Instance]], Optional[Sequence[Any]]
        ] = default_get_groups,
        batch_weight_key: str = "",
        retrain: bool = False,
    ) -> None:
        self.serialization_dir = serialization_dir
        self.dataset = dataset
        self.data_loader_builder = data_loader_builder
        self.model = model
        self.trainer_builder = trainer_builder
        self.cross_validator = cross_validator
        self.instances_labels_fn = instances_labels_fn
        self.instances_groups_fn = instances_groups_fn
        self.batch_weight_key = batch_weight_key
        self.retrain = retrain

    def run(self) -> Dict[str, Any]:
        metrics_by_fold = []

        n_splits = self.cross_validator.get_n_splits(
            self.dataset, labels_fn=self.instances_labels_fn, groups_fn=self.instances_groups_fn
        )

        for fold_index, (train_indices, test_indices) in enumerate(
            self.cross_validator(
                self.dataset, labels_fn=self.instances_labels_fn, groups_fn=self.instances_groups_fn
            )
        ):
            logger.info(f"Fold {fold_index}/{n_splits - 1}")

            serialization_dir = os.path.join(self.serialization_dir, f"fold_{fold_index}")
            if common_util.is_master():
                os.makedirs(serialization_dir, exist_ok=True)

            train_dataset = Subset(self.dataset, train_indices)
            # FIXME: `BucketBatchSampler` needs the dataset to have a vocab, so we workaround it:
            train_dataset.vocab = self.dataset.vocab
            test_dataset = Subset(self.dataset, test_indices)
            test_dataset.vocab = self.dataset.vocab

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

            if common_util.is_master():
                common_util.dump_metrics(
                    os.path.join(subtrainer._serialization_dir, "metrics.json"),
                    fold_metrics,
                    log=True,
                )

            metrics_by_fold.append(fold_metrics)

        metrics = {}

        for metric_key, fold_0_metric_value in metrics_by_fold[0].items():
            for fold_index, fold_metrics in enumerate(metrics_by_fold):
                metrics[f"fold{fold_index}_{metric_key}"] = fold_metrics[metric_key]
            if isinstance(fold_0_metric_value, float):
                average = Average()
                for fold_metrics in metrics_by_fold:
                    average(fold_metrics[metric_key])
                metrics[f"average_{metric_key}"] = average.get_metric()

        if self.retrain:
            data_loader = self.data_loader_builder.construct(dataset=self.dataset)

            # We don't need to pass `serialization_dir` and `local_rank` here, because they will
            # have been passed through the trainer by `from_params` already,
            # because they were keyword arguments to construct this class in the first place.
            trainer = self.trainer_builder.construct(model=self.model, data_loader=data_loader)

            trainer.train()

        return metrics

    def finish(self, metrics: Dict[str, Any]) -> None:
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
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
        instance_label_key: Optional[str] = None,
        instance_group_key: Optional[str] = None,
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

        instances_labels_fn = (
            (lambda instances: instances_get_key(instances, instance_label_key))
            if instance_label_key
            else default_get_labels
        )

        instances_groups_fn = (
            (lambda instances: instances_get_key(instances, instance_group_key))
            if instance_group_key
            else default_get_groups
        )

        return cls(
            serialization_dir=serialization_dir,
            dataset=dataset,
            model=model_,
            data_loader_builder=data_loader,
            trainer_builder=trainer,
            cross_validator=cross_validator,
            instances_labels_fn=instances_labels_fn,
            instances_groups_fn=instances_groups_fn,
            batch_weight_key=batch_weight_key,
            retrain=retrain,
        )
