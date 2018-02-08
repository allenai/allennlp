from typing import Dict
import argparse
import logging
import sys

from allennlp.commands.serve import Serve
from allennlp.commands.predict import Predict
from allennlp.commands.train import Train
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.make_vocab import MakeVocab
from allennlp.commands.elmo import Elmo
from allennlp.commands.subcommand import Subcommand
from allennlp.service.predictors import DemoModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Originally we were inconsistent about using hyphens and underscores
# in our flag names. We're switching to all-hyphens, but in the near-term
# we're still allowing the underscore_versions too, so as not to break any
# code. However, we'll use this lookup to log a warning if someone uses the
# old names.
DEPRECATED_FLAGS = {
        '--serialization_dir': '--serialization-dir',
        '--archive_file': '--archive-file',
        '--evaluation_data_file': '--evaluation-data-file',
        '--cuda_device': '--cuda-device',
        '--batch_size': '--batch-size'
}

def main(prog: str = None,
         model_overrides: Dict[str, DemoModel] = {},
         predictor_overrides: Dict[str, str] = {},
         subcommand_overrides: Dict[str, Subcommand] = {}) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes
    in the ``allennlp`` codebase. In particular, once you start creating your own
    ``Model`` s and so forth, it won't work for them. However, ``allennlp.run`` is
    simply a wrapper around this function. To use the command line interface with your
    own custom classes, just create your own script that imports all of the classes you want
    and then calls ``main()``.

    The default models for ``serve`` and the default predictors for ``predict`` are
    defined above. If you'd like to add more or use different ones, the
    ``model_overrides`` and ``predictor_overrides`` arguments will take precedence over the defaults.
    """
    # pylint: disable=dangerous-default-value

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]', prog=prog)
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "train": Train(),
            "evaluate": Evaluate(),
            "predict": Predict(predictor_overrides),
            "serve": Serve(model_overrides),
            "make-vocab": MakeVocab(),
            "elmo": Elmo(),

            # Superseded by overrides
            **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subcommand.add_subparser(name, subparsers)

    # Check and warn for deprecated args.
    for arg in sys.argv[1:]:
        if arg in DEPRECATED_FLAGS:
            logger.warning("Argument name %s is deprecated (and will likely go away at some point), "
                           "please use %s instead",
                           arg,
                           DEPRECATED_FLAGS[arg])

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
