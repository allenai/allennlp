import argparse

from overrides import overrides

from allennlp.commands import Subcommand


def do_nothing(_):
    pass


@Subcommand.register("d")
class D(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        subparser = parser.add_parser(self.name, description="fake", help="fake help")
        subparser.set_defaults(func=do_nothing)
        return subparser
