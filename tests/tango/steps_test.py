from tempfile import TemporaryDirectory
from typing import Iterable, Optional, Tuple

import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.tango.format import DillFormat, JsonFormat, _OPEN_FUNCTIONS
from allennlp.tango.hf_dataset import HuggingfaceDataset
from allennlp.tango.step import (
    Step,
    step_graph_from_params,
    MemoryStepCache,
    DirectoryStepCache,
    tango_dry_run,
)

import logging

from allennlp.tango.text_only import TextOnlyDataset

logging.basicConfig(level=logging.INFO)


def test_from_params():
    params = Params({"type": "hf_dataset", "dataset_name": "squad"})
    step = Step.from_params(params)
    result = step.result()
    assert "train" in result.splits


def test_from_params_wrong_type():
    params = Params({"type": "hf_dataset", "dataset_name": 1.1})
    with pytest.raises(ConfigurationError):
        Step.from_params(params)


def test_nested_steps():
    @Step.register("string")
    class StringStep(Step):
        def run(self, result: str) -> str:  # type: ignore
            return result

    params = Params({"type": "hf_dataset", "dataset_name": {"type": "string", "result": "squad"}})
    step = Step.from_params(params)
    assert "train" in step.result().splits


def test_nested_steps_wrong_type():
    @Step.register("float")
    class FloatStep(Step):
        def run(self, result: float) -> float:  # type: ignore
            return result

    params = Params({"type": "hf_dataset", "dataset_name": {"type": "float", "result": 1.1}})
    with pytest.raises(ConfigurationError):
        Step.from_params(params)


def test_make_step_graph():
    params = Params(
        {
            "steps": {
                "dataset": {"type": "hf_dataset", "dataset_name": "squad"},
                "dataset_text_only": {
                    "type": "text_only",
                    "input": {"type": "ref", "ref": "dataset"},
                    "fields_to_keep": ["context", "question"],
                },
            }
        }
    )
    step_graph = step_graph_from_params(params.pop("steps"))
    assert len(step_graph) == 2
    assert isinstance(step_graph["dataset"], HuggingfaceDataset)
    assert isinstance(step_graph["dataset_text_only"], TextOnlyDataset)
    assert step_graph["dataset_text_only"].kwargs["input"] == step_graph["dataset"]


@pytest.mark.parametrize("ordered_ascending", [True, False])
def test_make_step_graph_simple_ref(ordered_ascending: bool):
    params_as_dict_because_mypy_is_lame = {
        "dataset": {"type": "hf_dataset", "dataset_name": "squad"},
        "dataset_text_only": {
            "type": "text_only",
            "input": "dataset",
            "fields_to_keep": ["context", "question"],
        },
    }
    params_as_dict_because_mypy_is_lame = dict(
        sorted(params_as_dict_because_mypy_is_lame.items(), reverse=ordered_ascending)
    )
    params = Params({"steps": params_as_dict_because_mypy_is_lame})
    step_graph = step_graph_from_params(params.pop("steps"))
    assert len(step_graph) == 2
    assert isinstance(step_graph["dataset"], HuggingfaceDataset)
    assert isinstance(step_graph["dataset_text_only"], TextOnlyDataset)
    assert step_graph["dataset_text_only"].kwargs["input"] == step_graph["dataset"]


def test_make_step_graph_missing_step():
    params = Params(
        {
            "steps": {
                "dataset_text_only": {
                    "type": "text_only",
                    "input": "dataset",
                    "fields_to_keep": ["context", "question"],
                }
            }
        }
    )
    with pytest.raises(ConfigurationError):
        step_graph_from_params(params.pop("steps"))


@pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
def test_iterable_dill_format(compress: Optional[str]):
    r = (x + 1 for x in range(10))

    with TemporaryDirectory(prefix="test_iterable_dill_format-") as d:
        format = DillFormat[Iterable[int]](compress)
        format.write(r, d)
        r2 = format.read(d)
        assert [x + 1 for x in range(10)] == list(r2)


@pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
def test_iterable_json_format(compress: Optional[str]):
    r = (x + 1 for x in range(10))

    with TemporaryDirectory(prefix="test_iterable_json_format-") as d:
        format = JsonFormat[Iterable[int]](compress)
        format.write(r, d)
        r2 = format.read(d)
        assert [x + 1 for x in range(10)] == list(r2)


@pytest.mark.parametrize("step_cache_class", [MemoryStepCache, DirectoryStepCache])
def test_run_steps_programmatically(step_cache_class):
    from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
    from allennlp.tango.dataset import DatasetReaderAdapterStep
    from allennlp.tango import TrainingStep
    from allennlp.common import Lazy
    from allennlp.training.optimizers import AdamOptimizer
    from allennlp.tango.dataloader import BatchSizeDataLoader
    from allennlp.models import SimpleTagger
    from allennlp.tango import EvaluationStep

    dataset_step = DatasetReaderAdapterStep(
        reader=SequenceTaggingDatasetReader(),
        splits={
            "train": "test_fixtures/data/sequence_tagging.tsv",
            "validation": "test_fixtures/data/sequence_tagging.tsv",
        },
    )
    training_step = TrainingStep(
        model=Lazy(
            SimpleTagger,
            Params(
                {
                    "text_field_embedder": {
                        "token_embedders": {
                            "tokens": {
                                "type": "embedding",
                                "projection_dim": 2,
                                "pretrained_file": "test_fixtures/embeddings/glove.6B.100d.sample.txt.gz",
                                "embedding_dim": 100,
                                "trainable": True,
                            }
                        }
                    },
                    "encoder": {"type": "lstm", "input_size": 2, "hidden_size": 4, "num_layers": 1},
                }
            ),
        ),
        dataset=dataset_step,
        data_loader=Lazy(BatchSizeDataLoader, Params({"batch_size": 2})),
        optimizer=Lazy(AdamOptimizer),
    )
    evaluation_step = EvaluationStep(
        dataset=dataset_step, model=training_step, step_name="evaluation"
    )

    with TemporaryDirectory(prefix="test_run_steps_programmatically-") as d:
        if step_cache_class == DirectoryStepCache:
            cache = DirectoryStepCache(d)
        else:
            cache = step_cache_class()

        assert "random object" not in cache
        assert dataset_step not in cache
        assert training_step not in cache
        assert evaluation_step not in cache
        assert len(cache) == 0
        with pytest.raises(KeyError):
            _ = cache[evaluation_step]

        assert tango_dry_run(evaluation_step, cache) == [
            (dataset_step, False),
            (training_step, False),
            (evaluation_step, False),
        ]
        training_step.ensure_result(cache)
        assert tango_dry_run(evaluation_step, cache) == [
            (dataset_step, True),
            (training_step, True),
            (evaluation_step, False),
        ]

        assert "random object" not in cache
        assert dataset_step in cache
        assert training_step in cache
        assert evaluation_step not in cache
        assert len(cache) == 2
        with pytest.raises(KeyError):
            _ = cache[evaluation_step]


@pytest.mark.parametrize("deterministic", [True, False])
def test_random_seeds_are_initialized(deterministic: bool):
    class RandomNumberStep(Step[Tuple[int, int, int]]):
        DETERMINISTIC = deterministic
        CACHEABLE = False

        def run(self) -> Tuple[int, int, int]:  # type: ignore
            import random
            import numpy
            import torch

            return (
                random.randint(0, 2 ** 32),
                numpy.random.randint(0, 2 ** 32),
                torch.randint(2 ** 32, [1])[0].item(),
            )

    step1_result = RandomNumberStep().result()
    step2_result = RandomNumberStep().result()

    if deterministic:
        assert step1_result == step2_result
    else:
        assert step1_result != step2_result
