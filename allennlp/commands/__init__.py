import argparse

from allennlp.commands.serve import add_subparser as add_serve_subparser
from allennlp.commands.predict import add_subparser as add_predict_subparser
from allennlp.commands.train import add_subparser as add_train_subparser
from allennlp.commands.evaluate import add_subparser as add_evaluate_subparser
from allennlp.common.checks import ensure_pythonhashseed_set

def main(prog: str = None) -> None:
    ensure_pythonhashseed_set()

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]', prog=prog)
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    # Add sub-commands
    add_train_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_predict_subparser(subparsers)
    add_serve_subparser(subparsers)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
