import pytest

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.steps.examples import HuggingfaceDataset, TextOnlyDataset
from allennlp.steps.step import Step, step_graph_from_params

import logging

logging.basicConfig(level=logging.INFO)


def test_steps():
    dataset = HuggingfaceDataset(dataset_name="squad")
    flattened_dataset = TextOnlyDataset(input=dataset, fields_to_keep={"context", "question"})
    result = flattened_dataset.result()
    assert "train" in result.splits
    assert len(result.splits["train"]) == 2 * len(dataset.result().splits["train"])


def test_from_params():
    params = Params({"type": "huggingface_dataset", "dataset_name": "squad"})
    step = Step.from_params(params)
    result = step.result()
    assert "train" in result.splits


def test_from_params_wrong_type():
    params = Params({"type": "huggingface_dataset", "dataset_name": 1.1})
    with pytest.raises(ConfigurationError):
        Step.from_params(params)


def test_nested_steps():
    @Step.register("string")
    class StringStep(Step):
        def run(self, result: str) -> str:
            return result

    params = Params(
        {"type": "huggingface_dataset", "dataset_name": {"type": "string", "result": "squad"}}
    )
    step = Step.from_params(params)
    assert "train" in step.result().splits


def test_nested_steps_wrong_type():
    @Step.register("float")
    class FloatStep(Step):
        def run(self, result: float) -> float:
            return result

    params = Params(
        {"type": "huggingface_dataset", "dataset_name": {"type": "float", "result": 1.1}}
    )
    with pytest.raises(ConfigurationError):
        Step.from_params(params)


def test_make_step_graph():
    params = Params(
        {
            "steps": {
                "dataset": {"type": "huggingface_dataset", "dataset_name": "squad"},
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
    params = {
        "dataset": {"type": "huggingface_dataset", "dataset_name": "squad"},
        "dataset_text_only": {
            "type": "text_only",
            "input": "dataset",
            "fields_to_keep": ["context", "question"],
        },
    }
    params = dict(sorted(params.items(), reverse=ordered_ascending))
    params = Params({"steps": params})
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
