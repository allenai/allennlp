import os
from unittest import TestCase
from semantic_version import Version


def load_version():
    with open("allennlp/version.py", "r") as version_file:
        globals = {}
        exec(version_file.read(), globals)
        return Version(globals["VERSION"])


class TestVersion(TestCase):
    def test_default_suffix(self):
        if "ALLENNLP_VERSION_SUFFIX" in os.environ:
            del os.environ["ALLENNLP_VERSION_SUFFIX"]
        version = load_version()
        assert version.prerelease == ("unreleased",)
        assert version.build == ()

    def test_empty_suffix(self):
        os.environ["ALLENNLP_VERSION_SUFFIX"] = ""
        version = load_version()
        assert version.prerelease == ()
        assert version.build == ()

    def test_nightly_suffix(self):
        os.environ["ALLENNLP_VERSION_SUFFIX"] = "-dev20200212+c6147ad3"
        version = load_version()
        assert version.prerelease == ("dev20200212",)
        assert version.build == ("c6147ad3",)
