import pytest

from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.data.fields import FlagField


class TestFlagField(AllenNlpTestCase):
    def test_get_padding_lengths_returns_nothing(self):
        flag_field = FlagField(True)
        assert flag_field.get_padding_lengths() == {}

    def test_as_tensor_just_returns_value(self):
        for value in [True, 3.234, "this is a string"]:
            assert FlagField(value).as_tensor({}) == value

    def test_printing_doesnt_crash(self):
        flag = FlagField(True)
        print(flag)

    def test_batch_tensors_returns_single_value(self):
        value = True
        fields = [FlagField(value) for _ in range(5)]
        values = [field.as_tensor({}) for field in fields]
        batched_value = fields[0].batch_tensors(values)
        assert batched_value == value

    def test_batch_tensors_crashes_with_non_uniform_values(self):
        field = FlagField(True)
        with pytest.raises(ValueError):
            field.batch_tensors([True, False, True])

        with pytest.raises(ValueError):
            field.batch_tensors([1, 2, 3, 4])

        with pytest.raises(ValueError):
            field.batch_tensors(["different", "string", "flags"])
