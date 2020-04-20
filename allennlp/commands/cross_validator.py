from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

import sklearn.model_selection
from numpy.random.mtrand import RandomState
from overrides import overrides
from sklearn.model_selection import BaseCrossValidator

from allennlp.common import Registrable
from allennlp.data import Instance


def default_get_labels(instances: Sequence[Instance]) -> Optional[Sequence[Any]]:
    return [instance["label"].label for instance in instances] if "label" in instances[0] else None


def default_get_groups(instances: Sequence[Instance]) -> Optional[Sequence[Any]]:
    return [instance["group"].array for instance in instances] if "group" in instances[0] else None


class CrossValidator(Registrable, BaseCrossValidator):
    default_implementation = "k_fold"

    def __call__(
        self,
        instances: Sequence[Instance],
        labels_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_labels,
        groups_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_groups,
    ) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        labels, groups = self._labels_groups(instances, labels_fn=labels_fn, groups_fn=groups_fn)
        return super().split(instances, labels, groups=groups)

    @overrides
    def split(
        self,
        instances: Sequence[Instance],
        labels_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_labels,
        groups_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_groups,
    ) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        return self(instances, labels_fn=labels_fn, groups_fn=groups_fn)

    @overrides
    def get_n_splits(
        self,
        instances: Sequence[Instance],
        labels_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_labels,
        groups_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_groups,
    ) -> int:
        labels, groups = self._labels_groups(instances, labels_fn=labels_fn, groups_fn=groups_fn)
        return super().get_n_splits(instances, labels, groups=groups)

    @staticmethod
    def _labels_groups(
        instances: Sequence[Instance],
        labels_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_labels,
        groups_fn: Callable[[Sequence[Instance]], Optional[Sequence[Any]]] = default_get_groups,
    ) -> Tuple[Optional[Sequence[Any]], Optional[Sequence[Any]]]:
        return labels_fn(instances), groups_fn(instances)


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
