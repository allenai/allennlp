from allennlp.common.det_hash import det_hash, DetHashWithVersion


def test_normal_det_hash():
    class C:
        VERSION = 1

        def __init__(self, x: int):
            self.x = x

    c1_1 = C(10)
    c2_1 = C(10)
    c3_1 = C(20)
    assert det_hash(c1_1) == det_hash(c2_1)
    assert det_hash(c3_1) != det_hash(c2_1)

    class C:
        VERSION = 2

        def __init__(self, x: int):
            self.x = x

    c1_2 = C(10)
    c2_2 = C(10)
    c3_2 = C(20)
    assert det_hash(c1_2) == det_hash(c2_2)
    assert det_hash(c3_2) != det_hash(c2_2)
    assert det_hash(c1_2) == det_hash(c1_1)  # because the version isn't taken into account
    assert det_hash(c3_2) == det_hash(c3_1)  # because the version isn't taken into account


def test_versioned_det_hash():
    class C(DetHashWithVersion):
        VERSION = 1

        def __init__(self, x: int):
            self.x = x

    c1_1 = C(10)
    c2_1 = C(10)
    c3_1 = C(20)
    assert det_hash(c1_1) == det_hash(c2_1)
    assert det_hash(c3_1) != det_hash(c2_1)

    class C(DetHashWithVersion):
        VERSION = 2

        def __init__(self, x: int):
            self.x = x

    c1_2 = C(10)
    c2_2 = C(10)
    c3_2 = C(20)
    assert det_hash(c1_2) == det_hash(c2_2)
    assert det_hash(c3_2) != det_hash(c2_2)
    assert det_hash(c1_2) != det_hash(c1_1)  # because the version is taken into account
    assert det_hash(c3_2) != det_hash(c3_1)  # because the version is taken into account
