from datetime import timedelta
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

        x = util.sanitize({1, 2, 3})
        assert isinstance(x, list)
        assert len(x) == 3

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

    def test_sanitize_ptb_tokenized_string(self):
        def create_surrounding_test_case(start_ptb_token, end_ptb_token, start_token, end_token):
            return (
                "a {} b c {} d".format(start_ptb_token, end_ptb_token),
                "a {}b c{} d".format(start_token, end_token),
            )

        def create_fwd_token_test_case(fwd_token):
            return "a {} b".format(fwd_token), "a {}b".format(fwd_token)

        def create_backward_token_test_case(backward_token):
            return "a {} b".format(backward_token), "a{} b".format(backward_token)

        punct_forward = {"`", "$", "#"}
        punct_backward = {".", ",", "!", "?", ":", ";", "%", "'"}

        test_cases = [
            # Parentheses
            create_surrounding_test_case("-lrb-", "-rrb-", "(", ")"),
            create_surrounding_test_case("-lsb-", "-rsb-", "[", "]"),
            create_surrounding_test_case("-lcb-", "-rcb-", "{", "}"),
            # Parentheses don't have to match
            create_surrounding_test_case("-lsb-", "-rcb-", "[", "}"),
            # Also check that casing doesn't matter
            create_surrounding_test_case("-LsB-", "-rcB-", "[", "}"),
            # Quotes
            create_surrounding_test_case("``", "''", '"', '"'),
            # Start/end tokens
            create_surrounding_test_case("<s>", "</s>", "", ""),
            # Tokens that merge forward
            *[create_fwd_token_test_case(t) for t in punct_forward],
            # Tokens that merge backward
            *[create_backward_token_test_case(t) for t in punct_backward],
            # Merge tokens starting with ' backwards
            ("I 'm", "I'm"),
            # Merge tokens backwards when matching (n't or na) (special cases, parentheses behave in the same way)
            ("I do n't", "I don't"),
            ("gon na", "gonna"),
            # Also make sure casing is preserved
            ("gon NA", "gonNA"),
            # This is a no op
            ("A b C d", "A b C d"),
        ]

        for ptb_string, expected in test_cases:
            actual = util.sanitize_ptb_tokenized_string(ptb_string)
            assert actual == expected

    def test_cycle_iterator_function(self):
        global cycle_iterator_function_calls
        cycle_iterator_function_calls = 0

        def one_and_two():
            global cycle_iterator_function_calls
            cycle_iterator_function_calls += 1
            for i in [1, 2]:
                yield i

        iterator = iter(util.cycle_iterator_function(one_and_two))

        # Function calls should be lazy.
        assert cycle_iterator_function_calls == 0

        values = [next(iterator) for _ in range(5)]
        assert values == [1, 2, 1, 2, 1]
        # This is the difference between cycle_iterator_function and itertools.cycle.  We'd only see
        # 1 here with itertools.cycle.
        assert cycle_iterator_function_calls == 3


@pytest.mark.parametrize(
    "size, result",
    [
        (12, "12B"),
        (int(1.2 * 1024), "1.2K"),
        (12 * 1024, "12K"),
        (120 * 1024, "120K"),
        (int(1.2 * 1024 * 1024), "1.2M"),
        (12 * 1024 * 1024, "12M"),
        (120 * 1024 * 1024, "120M"),
        (int(1.2 * 1024 * 1024 * 1024), "1.2G"),
        (12 * 1024 * 1024 * 1024, "12G"),
    ],
)
def test_format_size(size: int, result: str):
    assert util.format_size(size) == result


@pytest.mark.parametrize(
    "td, result",
    [
        (timedelta(days=2, hours=3), "2 days"),
        (timedelta(days=1, hours=3), "1 day"),
        (timedelta(hours=3, minutes=12), "3 hours"),
        (timedelta(hours=1, minutes=12), "1 hour, 12 mins"),
        (timedelta(minutes=12), "12 mins"),
    ],
)
def test_format_timedelta(td: timedelta, result: str):
    assert util.format_timedelta(td) == result
