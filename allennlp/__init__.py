# On some systems this prevents the dreaded
# ImportError: dlopen: cannot load any more object with static TLS
import spacy, torch, numpy  # pylint: disable=multiple-imports
