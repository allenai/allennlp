import sys

import pytest

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


class TestCachedPathCommand(AllenNlpTestCase):
    def test_local_file(self, capsys):
        sys.argv = ["allennlp", "cached-path", "--cache-dir", str(self.TEST_DIR), "README.md"]
        main()
        captured = capsys.readouterr()
        assert "README.md" in captured.out

    def test_inspect_empty_cache(self, capsys):
        sys.argv = ["allennlp", "cached-path", "--cache-dir", str(self.TEST_DIR), "--inspect"]
        main()
        captured = capsys.readouterr()
        assert "Cached resources:" in captured.out
        assert "Total size: 0B" in captured.out

    def test_inspect_bad_options(self, capsys):
        sys.argv = [
            "allennlp",
            "cached-path",
            "--cache-dir",
            str(self.TEST_DIR),
            "--inspect",
            "--extract-archive",
        ]
        with pytest.raises(RuntimeError, match="--extract-archive"):
            main()
