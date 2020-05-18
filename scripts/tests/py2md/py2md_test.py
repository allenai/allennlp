from typing import Optional

import pytest

from allennlp.common.testing import AllenNlpTestCase
from scripts.py2md import py2md, Param, DocstringError


class TestPy2md(AllenNlpTestCase):
    def test_basic_example(self, capsys):
        py2md("scripts.tests.py2md.basic_example")
        captured = capsys.readouterr()

        with open(
            self.PROJECT_ROOT / "scripts" / "tests" / "py2md" / "basic_example_expected_output.md"
        ) as f:
            expected = f.read()

        assert captured.out.split("\n") == expected.split("\n")


@pytest.mark.parametrize(
    "line_in, line_out",
    [
        (
            "a : `int`, optional (default = `None`)",
            "- __a__ : `int`, optional (default = `None`) <br>",
        ),
        (
            "foo : `Tuple[int, ...]`, optional (default = `()`)",
            "- __foo__ : `Tuple[int, ...]`, optional (default = `()`) <br>",
        ),
        ("a : `int`, required", "- __a__ : `int` <br>"),
        ("a : `int`", "- __a__ : `int` <br>"),
        ("_a : `int`", "- __\\_a__ : `int` <br>"),
        ("a_ : `int`", "- __a\\___ : `int` <br>"),
    ],
)
def test_param_from_and_to_line(line_in: str, line_out: Optional[str]):
    param = Param.from_line(line_in)
    assert param is not None
    assert param.to_line() == line_out


@pytest.mark.parametrize(
    "line",
    [
        "a : `int`, optional (default = None)",
        "a : `int`, optional (default = `None)",
        "a : `int`, optional (default = None`)",
        "a : int",
        "a : `int",
        "a : int`",
    ],
)
def test_param_from_bad_line_raises(line: str):
    with pytest.raises(DocstringError):
        Param.from_line(line)
