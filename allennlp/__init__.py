# pylint: disable=wrong-import-position

# Make sure that allennlp is running in Python 3.6
import sys
if sys.version_info < (3, 6):
    raise RuntimeError("AllenNLP requires Python 3.6 or later")

# Disable FutureWarnings raised by h5py
# TODO(joelgrus): remove this (and pin requirements) when h5py 2.8.0 is available.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    # On some systems this prevents the dreaded
    # ImportError: dlopen: cannot load any more object with static TLS
    import spacy, torch, numpy  # pylint: disable=multiple-imports

except ModuleNotFoundError:
    print("Using AllenNLP requires the python packages Spacy, "
          "Pytorch and Numpy to be installed. Please see "
          "https://github.com/allenai/allennlp for installation instructions.")
    raise

from allennlp.version import VERSION as __version__
