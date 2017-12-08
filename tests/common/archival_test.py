# pylint: disable=invalid-name,protected-access,no-self-use

from typing import Dict

from allennlp.common import Params
from allennlp.common.archival import Archivable, collect
from allennlp.common.testing import AllenNlpTestCase

PARAMS = Params({
        "a": {
                "b": {
                        "filename": "my-file",
                        "c": {
                                "c_file": "my-other-file"
                        }
                }
        }
})


class A:
    def __init__(self, b: 'B') -> None:
        self.b = b

    @classmethod
    def from_params(cls, params: Params) -> 'A':
        b_params = params.pop("b")
        return cls(B.from_params(b_params))


class B(Archivable):
    def __init__(self, filename: str, c: 'C') -> None:
        self.filename = filename
        self.c_dict = {"here": c}

    @classmethod
    def from_params(cls, params: Params) -> 'B':
        filename = params.pop("filename")
        c_params = params.pop("c")
        c = C.from_params(c_params)

        instance = cls(filename, c)
        instance._param_history = params.history
        return instance

    def files_to_archive(self) -> Dict[str, str]:
        return {"filename": self.filename}

class C(Archivable):
    def __init__(self, c_file: str) -> None:
        self.c_file = c_file

    @classmethod
    def from_params(cls, params: Params) -> 'C':
        c_file = params.pop("c_file")
        instance = cls(c_file)
        instance._param_history = params.history
        return instance

    def files_to_archive(self) -> Dict[str, str]:
        return {"c_file": self.c_file}


class TestArchival(AllenNlpTestCase):
    def test_archival(self):
        a = A.from_params(PARAMS.pop("a"))

        collection = collect(a)

        assert collection == {
                "a.b.filename": "my-file",
                "a.b.c.c_file": "my-other-file"
        }
