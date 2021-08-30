import os
from tempfile import TemporaryDirectory

from allennlp.common.sqlite_sparse_sequence import SqliteSparseSequence


def test_sqlite_sparse_sequence():
    with TemporaryDirectory(prefix="test_sparse_sequence-") as temp_dir:
        s = SqliteSparseSequence(os.path.join(temp_dir, "test.sqlite"))
        assert len(s) == 0
        s.extend([])
        assert len(s) == 0
        s.append("one")
        assert len(s) == 1
        s.extend(["two", "three"])
        s.insert(1, "two")
        assert s[1] == "two"
        assert s.count("two") == 2
        ss = s[1:3]
        assert list(ss) == ["two", "two"]
        del s[1:3]
        assert len(s) == 2
        assert s[-1] == "three"
        s.clear()
        assert len(s) == 0
