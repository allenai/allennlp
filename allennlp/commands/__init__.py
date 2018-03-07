from typing import Dict
import argparse
import logging
import sys

from allennlp.commands.elmo import Elmo
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.fine_tune import FineTune
from allennlp.commands.make_vocab import MakeVocab
from allennlp.commands.predict import Predict
from allennlp.commands.serve import Serve
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import Train
from allennlp.service.predictors import DemoModel
from allennlp.common.util import import_submodules

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main(prog: str = None,
         model_overrides: Dict[str, DemoModel] = {},
         predictor_overrides: Dict[str, str] = {},
         subcommand_overrides: Dict[str, Subcommand] = {}) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag available for most commands.
    """
    # pylint: disable=dangerous-default-value

    # TODO(mattg): document and/or remove the `predictor_overrides` and `model_overrides` commands.
    # The `--predictor` option for the `predict` command largely removes the need for
    # `predictor_overrides`, and I think the simple server largely removes the need for
    # `model_overrides`, and maybe the whole `serve` command as a public API (we only need that
    # path for demo.allennlp.org, and it's not likely anyone else would host that particular demo).

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s', prog=prog)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "train": Train(),
            "evaluate": Evaluate(),
            "predict": Predict(predictor_overrides),
            "serve": Serve(model_overrides),
            "make-vocab": MakeVocab(),
            "elmo": Elmo(),
            "fine-tune": FineTune(),

            # Superseded by overrides
            **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')

    args = parser.parse_args()

    # Import any additional modules needed (to register custom classes).
    for package_name in args.include_package:
        import_submodules(package_name)

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
