from typing import Sequence
from common.checks import ensure_pythonhashseed_set
import argparse

import allennlp.commands.serve as serve
import allennlp.commands.predict as predict
import allennlp.commands.train as train
import allennlp.commands.evaluate as evaluate

def main(raw_args: Sequence[str]) -> None:
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
