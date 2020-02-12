import os

_MAJOR = "0"
_MINOR = "9"

# We check the environment for the revision here, because it
# enables us to set it to a git sha when doing automated nightly releases.
_REVISION = os.environ.get("ALLENNLP_REVISION", "1-unreleased")

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}".format(_MAJOR, _MINOR, _REVISION)
