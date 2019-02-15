# pylint: disable=no-self-use,invalid-name,too-many-public-methods,protected-access
from typing import Dict, Optional, List, Tuple, Set

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

    def test_no_constructor(self):
        params = Params({"type": "just_spaces"})

        WordSplitter.from_params(params)

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
