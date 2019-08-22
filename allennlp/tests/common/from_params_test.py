# pylint: disable=no-self-use,invalid-name,too-many-public-methods,protected-access
from typing import Dict, Optional, List, Set, Tuple, Union

import pytest
import torch

from allennlp.common import Params
from allennlp.common.from_params import FromParams, takes_arg, remove_optional, create_kwargs
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.common.checks import ConfigurationError

class MyClass(FromParams):
    def __init__(self, my_int: int, my_bool: bool = False) -> None:
        self.my_int = my_int
        self.my_bool = my_bool


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

    def test_from_params(self):
        my_class = MyClass.from_params(Params({"my_int": 10}), my_bool=True)

        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_create_kwargs(self):
        kwargs = create_kwargs(MyClass,
                               Params({'my_int': 5}),
                               my_bool=True,
                               my_float=4.4)

        # my_float should not be included because it's not a param of the MyClass constructor
        assert kwargs == {
                "my_int": 5,
                "my_bool": True
        }

    def test_extras(self):
        # pylint: disable=unused-variable,arguments-differ
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

        @A.register("c")
        class C(A):
            def __init__(self, size: int, name: str) -> None:
                self.size = size
                self.name = name

            # custom from params
            @classmethod
            def from_params(cls, params: Params, size: int) -> 'C':  # type: ignore
                name = params.pop('name')
                return cls(size=size, name=name)


        # Check that extras get passed, even though A doesn't need them.
        params = Params({"type": "b", "size": 10})
        b = A.from_params(params, name="extra")

        assert b.name == "extra"
        assert b.size == 10

        # Check that extra extras don't get passed.
        params = Params({"type": "b", "size": 10})
        b = A.from_params(params, name="extra", unwanted=True)

        assert b.name == "extra"
        assert b.size == 10

        # Now the same with a custom from_params.
        params = Params({"type": "c", "name": "extra_c"})
        c = A.from_params(params, size=20)
        assert c.name == "extra_c"
        assert c.size == 20

        # Check that extra extras don't get passed.
        params = Params({"type": "c", "name": "extra_c"})
        c = A.from_params(params, size=20, unwanted=True)

        assert c.name == "extra_c"
        assert c.size == 20

    def test_extras_for_custom_classes(self):
        # pylint: disable=unused-variable,arguments-differ
        from allennlp.common.registrable import Registrable

        class BaseClass(Registrable):
            pass

        class BaseClass2(Registrable):
            pass

        @BaseClass.register("A")
        class A(BaseClass):
            def __init__(self, a: int, b: int, val: str) -> None:
                self.a = a
                self.b = b
                self.val = val

            def __hash__(self):
                return self.b

            def __eq__(self, other):
                return self.b == other.b

            @classmethod
            def from_params(cls, params: Params, a: int) -> 'A':  # type: ignore
                # A custom from params
                b = params.pop_int("b")
                val = params.pop("val", "C")
                params.assert_empty(cls.__name__)
                return cls(a=a, b=b, val=val)

        @BaseClass2.register("B")
        class B(BaseClass2):
            def __init__(self, c: int, b: int) -> None:
                self.c = c
                self.b = b

            @classmethod
            def from_params(cls, params: Params, c: int) -> 'B':  # type: ignore
                b = params.pop_int("b")
                params.assert_empty(cls.__name__)
                return cls(c=c, b=b)

        @BaseClass.register("E")
        class E(BaseClass):
            def __init__(self, m: int, n: int) -> None:
                self.m = m
                self.n = n

            @classmethod
            def from_params(cls, params: Params, **extras2) -> 'E':  # type: ignore
                m = params.pop_int("m")
                params.assert_empty(cls.__name__)
                n = extras2["n"]
                return cls(m=m, n=n)

        class C(object):
            pass

        @BaseClass.register("D")
        class D(BaseClass):
            def __init__(self, arg1: List[BaseClass],
                         arg2: Tuple[BaseClass, BaseClass2],
                         arg3: Dict[str, BaseClass], arg4: Set[BaseClass],
                         arg5: List[BaseClass]) -> None:
                self.arg1 = arg1
                self.arg2 = arg2
                self.arg3 = arg3
                self.arg4 = arg4
                self.arg5 = arg5

        vals = [1, 2, 3]
        params = Params({"type": "D",
                         "arg1": [{"type": "A", "b": vals[0]},
                                  {"type": "A", "b": vals[1]},
                                  {"type": "A", "b": vals[2]}],
                         "arg2": [{"type": "A", "b": vals[0]},
                                  {"type": "B", "b": vals[0]}],
                         "arg3": {"class_1": {"type": "A", "b": vals[0]},
                                  "class_2": {"type": "A", "b": vals[1]}},
                         "arg4": [{"type": "A", "b": vals[0], "val": "M"},
                                  {"type": "A", "b": vals[1], "val": "N"},
                                  {"type": "A", "b": vals[1], "val": "N"}],
                         "arg5": [{"type": "E", "m": 9}]})
        extra = C()
        tval1 = 5
        tval2 = 6
        d = BaseClass.from_params(params=params, extra=extra, a=tval1, c=tval2, n=10)

        # Tests for List Parameters
        assert len(d.arg1) == len(vals)
        assert isinstance(d.arg1, list)
        assert isinstance(d.arg1[0], A)
        assert all([x.b == y for x, y in zip(d.arg1, vals)])
        assert all([x.a == tval1 for x in d.arg1])

        # Tests for Tuple
        assert isinstance(d.arg2, tuple)
        assert isinstance(d.arg2[0], A)
        assert isinstance(d.arg2[1], B)
        assert d.arg2[0].a == tval1
        assert d.arg2[1].c == tval2
        assert d.arg2[0].b == d.arg2[1].b == vals[0]

        # Tests for Dict
        assert isinstance(d.arg3, dict)
        assert isinstance(d.arg3["class_1"], A)
        assert d.arg3["class_1"].a == d.arg3["class_2"].a == tval1
        assert d.arg3["class_1"].b == vals[0]
        assert d.arg3["class_2"].b == vals[1]

        # Tests for Set
        assert isinstance(d.arg4, set)
        assert len(d.arg4) == 2
        assert any(x.val == "M" for x in d.arg4)
        assert any(x.val == "N" for x in d.arg4)

        # Tests for custom extras parameters
        assert isinstance(d.arg5, list)
        assert isinstance(d.arg5[0], E)
        assert d.arg5[0].m == 9
        assert d.arg5[0].n == 10

    def test_no_constructor(self):
        params = Params({"type": "just_spaces"})

        WordSplitter.from_params(params)

    def test_union(self):
        class A(FromParams):
            def __init__(self, a: Union[int, List[int]]) -> None:
                self.a = a

        class B(FromParams):
            def __init__(self, b: Union[A, List[A]]) -> None:
                # Really you would want to be sure that `self.b` has a consistent type, but for
                # this test we'll ignore that.
                self.b = b

        class C(FromParams):
            def __init__(self, c: Union[A, B, Dict[str, A]]) -> None:
                # Really you would want to be sure that `self.c` has a consistent type, but for
                # this test we'll ignore that.
                self.c = c

        params = Params({'a': 3})
        a = A.from_params(params)
        assert a.a == 3

        params = Params({'a': [3, 4, 5]})
        a = A.from_params(params)
        assert a.a == [3, 4, 5]

        params = Params({'b': {'a': 3}})
        b = B.from_params(params)
        assert isinstance(b.b, A)
        assert b.b.a == 3

        params = Params({'b': [{'a': 3}, {'a': [4, 5]}]})
        b = B.from_params(params)
        assert isinstance(b.b, list)
        assert b.b[0].a == 3
        assert b.b[1].a == [4, 5]

        # This is a contrived, ugly example (why would you want to duplicate names in a nested
        # structure like this??), but it demonstrates a potential bug when dealing with mutatable
        # parameters.  If you're not careful about keeping the parameters un-mutated in two
        # separate places, you'll end up with a B, or with a dict that's missing the 'b' key.
        params = Params({'c': {'a': {'a': 3}, 'b': {'a': [4, 5]}}})
        c = C.from_params(params)
        assert isinstance(c.c, dict)
        assert c.c['a'].a == 3
        assert c.c['b'].a == [4, 5]

    def test_dict(self):
        # pylint: disable=unused-variable
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Dict[str, A]) -> None:
                self.items = items

        params = Params({"type": "d", "items": {"first": {"type": "b", "size": 1},
                                                "second": {"type": "b", "size": 2}}})
        d = C.from_params(params)

        assert isinstance(d.items, dict)
        assert len(d.items) == 2
        assert all(isinstance(key, str) for key in d.items.keys())
        assert all(isinstance(value, B) for value in d.items.values())
        assert d.items["first"].size == 1
        assert d.items["second"].size == 2

    def test_dict_not_params(self):
        class A(FromParams):
            def __init__(self, counts: Dict[str, int]) -> None:
                self.counts = counts

        params = Params({"counts": {"a": 10, "b": 20}})
        a = A.from_params(params)

        assert isinstance(a.counts, dict)
        assert not isinstance(a.counts, Params)

    def test_list(self):
        # pylint: disable=unused-variable
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: List[A]) -> None:
                self.items = items

        params = Params({"type": "d", "items": [{"type": "b", "size": 1}, {"type": "b", "size": 2}]})
        d = C.from_params(params)

        assert isinstance(d.items, list)
        assert len(d.items) == 2
        assert all(isinstance(item, B) for item in d.items)
        assert d.items[0].size == 1
        assert d.items[1].size == 2

    def test_tuple(self):
        # pylint: disable=unused-variable
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        @A.register("b")
        class B(A):
            def __init__(self, size: int) -> None:
                self.size = size

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, name: str) -> None:
                self.name = name

        class E(Registrable):
            pass

        @E.register("f")
        class F(E):
            def __init__(self, items: Tuple[A, C]) -> None:
                self.items = items

        params = Params({"type": "f", "items": [{"type": "b", "size": 1}, {"type": "d", "name": "item2"}]})
        f = E.from_params(params)

        assert isinstance(f.items, tuple)
        assert len(f.items) == 2
        assert isinstance(f.items[0], B)
        assert isinstance(f.items[1], D)
        assert f.items[0].size == 1
        assert f.items[1].name == "item2"

    def test_set(self):
        # pylint: disable=unused-variable
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            def __init__(self, name: str) -> None:
                self.name = name

            def __eq__(self, other):
                return self.name == other.name

            def __hash__(self):
                return hash(self.name)

        @A.register("b")
        class B(A):
            pass

        class C(Registrable):
            pass

        @C.register("d")
        class D(C):
            def __init__(self, items: Set[A]) -> None:
                self.items = items

        params = Params({"type": "d", "items": [{"type": "b", "name": "item1"},
                                                {"type": "b", "name": "item2"},
                                                {"type": "b", "name": "item2"}]})
        d = C.from_params(params)

        assert isinstance(d.items, set)
        assert len(d.items) == 2
        assert all(isinstance(item, B) for item in d.items)
        assert any(item.name == "item1" for item in d.items)
        assert any(item.name == "item2" for item in d.items)

    def test_transferring_of_modules(self):

        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model

        config_file = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        model_params = Params.from_file(config_file).pop("model").as_dict(quiet=True)

        # Override only text_field_embedder (freeze) and attend_feedforward params (tunable)
        model_params["text_field_embedder"] = {
                "_pretrained": {
                        "archive_file": model_archive,
                        "module_path": "_text_field_embedder",
                        "freeze": True
                }
        }
        model_params["attend_feedforward"] = {
                "_pretrained": {
                        "archive_file": model_archive,
                        "module_path": "_attend_feedforward._module",
                        "freeze": False
                }
        }

        transfer_model = Model.from_params(vocab=trained_model.vocab,
                                           params=Params(model_params))

        # TextFieldEmbedder and AttendFeedforward parameters should be transferred
        for trained_parameter, transfer_parameter in zip(trained_model._text_field_embedder.parameters(),
                                                         transfer_model._text_field_embedder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        for trained_parameter, transfer_parameter in zip(trained_model._attend_feedforward.parameters(),
                                                         transfer_model._attend_feedforward.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        # Any other module's parameters shouldn't be same (eg. compare_feedforward)
        for trained_parameter, transfer_parameter in zip(trained_model._compare_feedforward.parameters(),
                                                         transfer_model._compare_feedforward.parameters()):
            assert torch.all(trained_parameter != transfer_parameter)

        # TextFieldEmbedder should have requires_grad Off
        for parameter in transfer_model._text_field_embedder.parameters():
            assert not parameter.requires_grad

        # # AttendFeedforward should have requires_grad On
        for parameter in transfer_model._attend_feedforward.parameters():
            assert parameter.requires_grad

    def test_transferring_of_modules_ensures_type_consistency(self):

        model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        trained_model = load_archive(model_archive).model

        config_file = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'experiment.json')
        model_params = Params.from_file(config_file).pop("model").as_dict(quiet=True)

        # Override only text_field_embedder and make it load AttendFeedForward
        model_params["text_field_embedder"] = {
                "_pretrained": {
                        "archive_file": model_archive,
                        "module_path": "_attend_feedforward._module"
                }
        }
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=trained_model.vocab,
                              params=Params(model_params))
