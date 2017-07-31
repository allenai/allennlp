from typing import Sequence
import argparse
import logging
import sys

from allennlp.commands.bulk import add_bulk_subparser
from allennlp.common.params import PARAMETER

# disable parameter logging
logging.disable(PARAMETER)

def main(raw_args: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]')
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    # Add sub-commands
    add_bulk_subparser(subparsers)

    args = parser.parse_args(raw_args)
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main(sys.argv)
