import re

import pytest

from allennlp.version import VERSION


# Regex to check that the current version set in `allennlp.version` adheres to
# PEP 440, as well as some of our own internal conventions, such as the `.dev`
# suffix being used only for nightly builds.
# 0.0.0rc0.post0.dev20200424
VALID_VERSION_RE = re.compile(
    r"^"
    r"(0|[1-9]\d*)"  # major
    r"\.(0|[1-9]\d*)"  # minor
    r"\.(0|[1-9]\d*)"  # patch
    r"(rc(0|[1-9]\d*))?"  # patch suffix
    r"(\.post(0|[1-9]\d*))?"  # [.postN]
    r"(\.dev2020[0-9]{4})?"  # [.devDATE]
    r"$"
)


def is_valid(version: str) -> bool:
    return VALID_VERSION_RE.match(version) is not None


@pytest.mark.parametrize(
    "version, valid",
    [
        # Valid versions:
        ("1.0.0", True),
        ("1.0.0rc3", True),
        ("1.0.0.post0", True),
        ("1.0.0.post1", True),
        ("1.0.0rc3.post0", True),
        ("1.0.0rc3.post0.dev20200424", True),
        # Invalid versions:
        ("1.0.0.rc3", False),
        ("1.0.0rc01", False),
        ("1.0.0rc3.dev2020424", False),
    ],
)
def test_is_valid_helper(version: str, valid: bool):
    assert is_valid(version) is valid


def test_version():
    """
    Ensures current version is consistent with our conventions.
    """
    assert is_valid(VERSION)
