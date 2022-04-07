# We get a lot of these spurious warnings,
# see https://github.com/ContinuumIO/anaconda-issues/issues/6678
import warnings  # noqa

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

try:
    # On some systems this prevents the dreaded
    # ImportError: dlopen: cannot load any more object with static TLS
    import transformers, spacy, torch, numpy  # noqa

except ModuleNotFoundError:
    print(
        "Using AllenNLP requires the python packages Spacy, "
        "Pytorch and Numpy to be installed. Please see "
        "https://github.com/allenai/allennlp for installation instructions."
    )
    raise

from allennlp.version import VERSION as __version__  # noqa
