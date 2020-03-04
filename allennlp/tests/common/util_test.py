import sys
from collections import OrderedDict

import pytest
import torch

from allennlp.common import util
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import push_python_path


class Unsanitizable:
    pass


class Sanitizable:
    def to_json(self):
        return {"sanitizable": True}


class TestCommonUtils(AllenNlpTestCase):
    def test_group_by_count(self):
        assert util.group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 20) == [
            [1, 2, 3],
            [4, 5, 6],
            [7, 20, 20],
        ]

    def test_lazy_groups_of(self):
        xs = [1, 2, 3, 4, 5, 6, 7]
        groups = util.lazy_groups_of(iter(xs), group_size=3)
        assert next(groups) == [1, 2, 3]
        assert next(groups) == [4, 5, 6]
        assert next(groups) == [7]
        with pytest.raises(StopIteration):
            _ = next(groups)

    def test_pad_sequence_to_length(self):
        assert util.pad_sequence_to_length([1, 2, 3], 5) == [1, 2, 3, 0, 0]
        assert util.pad_sequence_to_length([1, 2, 3], 5, default_value=lambda: 2) == [1, 2, 3, 2, 2]
        assert util.pad_sequence_to_length([1, 2, 3], 5, padding_on_right=False) == [0, 0, 1, 2, 3]

    def test_namespace_match(self):
        assert util.namespace_match("*tags", "tags")
        assert util.namespace_match("*tags", "passage_tags")
        assert util.namespace_match("*tags", "question_tags")
        assert util.namespace_match("tokens", "tokens")
        assert not util.namespace_match("tokens", "stemmed_tokens")

    def test_sanitize(self):
        assert util.sanitize(torch.Tensor([1, 2])) == [1, 2]
        assert util.sanitize(torch.LongTensor([1, 2])) == [1, 2]

        with pytest.raises(ValueError):
            util.sanitize(Unsanitizable())

        assert util.sanitize(Sanitizable()) == {"sanitizable": True}

    def test_import_submodules(self):
        (self.TEST_DIR / "mymodule").mkdir()
        (self.TEST_DIR / "mymodule" / "__init__.py").touch()
        (self.TEST_DIR / "mymodule" / "submodule").mkdir()
        (self.TEST_DIR / "mymodule" / "submodule" / "__init__.py").touch()
        (self.TEST_DIR / "mymodule" / "submodule" / "subsubmodule.py").touch()

        with push_python_path(self.TEST_DIR):
            assert "mymodule" not in sys.modules
            assert "mymodule.submodule" not in sys.modules

            util.import_module_and_submodules("mymodule")

            assert "mymodule" in sys.modules
            assert "mymodule.submodule" in sys.modules
            assert "mymodule.submodule.subsubmodule" in sys.modules

    def test_get_frozen_and_tunable_parameter_names(self):
        model = torch.nn.Sequential(
            OrderedDict([("conv", torch.nn.Conv1d(5, 5, 5)), ("linear", torch.nn.Linear(5, 10))])
        )
        named_parameters = dict(model.named_parameters())
        named_parameters["linear.weight"].requires_grad_(False)
        named_parameters["linear.bias"].requires_grad_(False)
        (
            frozen_parameter_names,
            tunable_parameter_names,
        ) = util.get_frozen_and_tunable_parameter_names(model)
        assert set(frozen_parameter_names) == {"linear.weight", "linear.bias"}
        assert set(tunable_parameter_names) == {"conv.weight", "conv.bias"}
