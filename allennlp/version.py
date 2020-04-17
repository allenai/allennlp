import os

_MAJOR = "1"
_MINOR = "0"
# On master and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "0"
# For pre-release and build metadata. In an official release this must be the
# empty string. On master we will default to "-unreleased" while in our nightly
# builds this will have the syntax ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = os.environ.get("ALLENNLP_VERSION_SUFFIX", ".rc2-unreleased")

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)
