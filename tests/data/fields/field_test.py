from allennlp.data.fields import Field


def test_eq_with_inheritance():
    class SubField(Field):

        __slots__ = ["a"]

        def __init__(self, a):
            self.a = a

    class SubSubField(SubField):

        __slots__ = ["b"]

        def __init__(self, a, b):
            super().__init__(a)
            self.b = b

    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)

    assert SubSubField(1, 2) == SubSubField(1, 2)
    assert SubSubField(1, 2) != SubSubField(1, 1)
    assert SubSubField(1, 2) != SubSubField(2, 2)
