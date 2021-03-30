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

    class SubSubSubField(SubSubField):

        __slots__ = ["c"]

        def __init__(self, a, b, c):
            super().__init__(a, b)
            self.c = c

    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)

    assert SubSubField(1, 2) == SubSubField(1, 2)
    assert SubSubField(1, 2) != SubSubField(1, 1)
    assert SubSubField(1, 2) != SubSubField(2, 2)

    assert SubSubSubField(1, 2, 3) == SubSubSubField(1, 2, 3)
    assert SubSubSubField(1, 2, 3) != SubSubSubField(0, 2, 3)


def test_eq_with_inheritance_for_non_slots_field():
    class SubField(Field):
        def __init__(self, a):
            self.a = a

    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)


def test_eq_with_inheritance_for_mixed_field():
    class SubField(Field):

        __slots__ = ["a"]

        def __init__(self, a):
            self.a = a

    class SubSubField(SubField):
        def __init__(self, a, b):
            super().__init__(a)
            self.b = b

    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)

    assert SubSubField(1, 2) == SubSubField(1, 2)
    assert SubSubField(1, 2) != SubSubField(1, 1)
    assert SubSubField(1, 2) != SubSubField(2, 2)
