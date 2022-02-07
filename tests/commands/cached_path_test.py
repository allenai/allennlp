import sys

import pytest

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


class TestCachedPathCommand(AllenNlpTestCase):
    def test_local_file(self, capsys):
        sys.argv = [
            "allennlp",
            "cached-path",
            "--cache-dir",
            str(self.TEST_DIR),
            str(self.PROJECT_ROOT_FALLBACK / "README.md"),
        ]
        main()
        captured = capsys.readouterr()
        assert "README.md" in captured.out

    def test_inspect_empty_cache(self, capsys):
        sys.argv = ["allennlp", "cached-path", "--cache-dir", str(self.TEST_DIR), "--inspect"]
        main()
        captured = capsys.readouterr()
        assert "Cached resources:" in captured.out
        assert "Total size: 0B" in captured.out

    def test_inspect_with_bad_options(self, capsys):
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

    def test_remove_with_bad_options(self, capsys):
        sys.argv = [
            "allennlp",
            "cached-path",
            "--cache-dir",
            str(self.TEST_DIR),
            "--remove",
            "--extract-archive",
            "*",
        ]
        with pytest.raises(RuntimeError, match="--extract-archive"):
            main()

    def test_remove_with_missing_positionals(self, capsys):
        sys.argv = [
            "allennlp",
            "cached-path",
            "--cache-dir",
            str(self.TEST_DIR),
            "--remove",
        ]
        with pytest.raises(RuntimeError, match="Missing positional"):
            main()

    def test_remove_empty_cache(self, capsys):
        sys.argv = [
            "allennlp",
            "cached-path",
            "--cache-dir",
            str(self.TEST_DIR),
            "--remove",
            "*",
        ]
        main()
        captured = capsys.readouterr()
        assert "Reclaimed 0B of space" in captured.out
