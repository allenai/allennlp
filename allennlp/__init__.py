# pylint: disable=wrong-import-position

# Make sure that allennlp is running in Python 3.6
import sys
if sys.version_info < (3, 6):
    raise RuntimeError("AllenNLP requires Python 3.6 or later")

# On some systems this prevents the dreaded
# ImportError: dlopen: cannot load any more object with static TLS
import spacy, torch, numpy  # pylint: disable=multiple-imports

from allennlp.version import VERSION as __version__
