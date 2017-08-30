"""
The ``main`` function is the entry point to all of
AllenNLP's command line features. Any executable
that calls ``main()`` automatically inherits all
of its child commands. In particular, once you create
your own models, you can create a script that imports
them and then calls ``main()``, and then use that script
to train and evaluate them from the command line.
"""

from typing import Sequence
import argparse

import allennlp.commands.serve as serve
import allennlp.commands.predict as predict
import allennlp.commands.train as train
import allennlp.commands.evaluate as evaluate
from allennlp.common.checks import ensure_pythonhashseed_set

def main(raw_args: Sequence[str]) -> None:
    """
    Any executable that calls this function automatically inherits
    all of its subcommands.
    """
    ensure_pythonhashseed_set()

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]')
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    # Add sub-commands
    predict.add_subparser(subparsers)
    train.add_subparser(subparsers)
    serve.add_subparser(subparsers)
    evaluate.add_subparser(subparsers)

    args = parser.parse_args(raw_args)

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
