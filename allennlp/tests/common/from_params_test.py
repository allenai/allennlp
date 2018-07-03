# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from typing import Dict, Optional

from allennlp.common.from_params import takes_arg, remove_optional
from allennlp.common.testing import AllenNlpTestCase


class TestFromParams(AllenNlpTestCase):
    def test_takes_arg(self):
        def bare_function(some_input: int) -> int:
            return some_input + 1

        assert takes_arg(bare_function, 'some_input')
        assert not takes_arg(bare_function, 'some_other_input')

        class SomeClass:
            total = 0

            def __init__(self, constructor_param: str) -> None:
                self.constructor_param = constructor_param

            def check_param(self, check: str) -> bool:
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total: int) -> None:
                cls.total = new_total

        assert takes_arg(SomeClass, 'self')
        assert takes_arg(SomeClass, 'constructor_param')
        assert not takes_arg(SomeClass, 'check')

        assert takes_arg(SomeClass.check_param, 'check')
        assert not takes_arg(SomeClass.check_param, 'other_check')

        assert takes_arg(SomeClass.set_total, 'new_total')
        assert not takes_arg(SomeClass.set_total, 'total')

    def test_remove_optional(self):
        optional_type = Optional[Dict[str, str]]
        bare_type = remove_optional(optional_type)
        bare_bare_type = remove_optional(bare_type)

        assert bare_type == Dict[str, str]
        assert bare_bare_type == Dict[str, str]

        assert remove_optional(Optional[str]) == str
        assert remove_optional(str) == str
