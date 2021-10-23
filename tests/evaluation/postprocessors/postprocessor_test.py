from typing import Iterator, List, Dict
import torch
import pytest
import json

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data.data_loaders import TensorDict
from allennlp.evaluation import Postprocessor
from allennlp.evaluation.postprocessors import SimplePostprocessor


class DummyDataLoader:
    def __init__(self, outputs: List[TensorDict]) -> None:
        super().__init__()
        self._outputs = outputs

    def __iter__(self) -> Iterator[TensorDict]:
        yield from self._outputs

    def __len__(self):
        return len(self._outputs)

    def set_target_device(self, _):
        pass


class TestPostprocessor(AllenNlpTestCase):
    def setup_method(self):
        super(TestPostprocessor, self).setup_method()
        self.postprocessor = Postprocessor.from_params(Params({}))

    def test_postprocessor_default_implementation(self):
        assert self.postprocessor.to_params().params == {"type": "simple"}
        assert isinstance(self.postprocessor, SimplePostprocessor)

    @pytest.mark.parametrize("batch", [
        {
            "Do you want ants?": "Because that's how you get ants.",
            "testing"          : torch.tensor([[1, 2, 3]])
        },
        {},
        None
    ], ids=["TestBatch", "EmptyBatch", "None"])
    @pytest.mark.parametrize("output_dict", [
        {"You're": ["Not", [["My"]], "Supervisor"]},
        {},
        None
    ], ids=["TestOutput", "EmptyOutput", "None"])
    def test_simple_postprocessor_call(self, batch, output_dict):
        data_loader = DummyDataLoader([])
        if batch is None or output_dict is None:
            with pytest.raises(ValueError):
                self.postprocessor(batch, output_dict, data_loader)  # type: ignore
            return

        expected = json.dumps(sanitize({**batch, **output_dict}))
        result = self.postprocessor(batch, output_dict, data_loader)  # type: ignore
        assert result == expected
