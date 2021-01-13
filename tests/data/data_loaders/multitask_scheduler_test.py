from allennlp.data.data_loaders.multitask_scheduler import (
    RoundRobinScheduler,
    HomogeneousRoundRobinScheduler,
)


class RoundRobinSchedulerTest:
    def test_order_instances(self):
        scheduler = RoundRobinScheduler()
        epoch_instances = {
            "a": [1] * 5,
            "b": [2] * 3,
        }
        flattened = scheduler.batch_instances(epoch_instances)
        assert list(flattened) == [
            ("a", 1),
            ("b", 2),
            ("a", 1),
            ("b", 2),
            ("a", 1),
            ("b", 2),
            ("a", 1),
            ("a", 1),
        ]


class HomogeneousRoundRobinSchedulerTest:
    def test_order_instances(self):
        scheduler = HomogeneousRoundRobinScheduler({"a": 2, "b": 3})
        epoch_instances = {
            "a": [1] * 9,
            "b": [2] * 9,
        }
        flattened = scheduler.batch_instances(epoch_instances)
        assert list(flattened) == [
            ("a", 1),
            ("a", 1),
            ("b", 2),
            ("b", 2),
            ("b", 2),
            ("a", 1),
            ("a", 1),
            ("b", 2),
            ("b", 2),
            ("b", 2),
            ("a", 1),
            ("a", 1),
            ("b", 2),
            ("b", 2),
            ("b", 2),
            ("a", 1),
            ("a", 1),
            ("a", 1),
        ]
