from tempfile import TemporaryDirectory
from typing import Iterable, Optional

import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.tango.format import DillFormat
from allennlp.tango.hf_dataset import HuggingfaceDataset
from allennlp.tango.step import Step, step_graph_from_params

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


@pytest.mark.parametrize("compress", DillFormat.OPEN_FUNCTIONS.keys())
def test_iterable_dill_format(compress: Optional[str]):
    r = (x + 1 for x in range(10))

    with TemporaryDirectory(prefix="test_iterable_dill_format-") as d:
        format = DillFormat[Iterable[int]](compress)
        format.write(r, d)
        r2 = format.read(d)
        assert [x + 1 for x in range(10)] == list(r2)
