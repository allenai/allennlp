import argparse

from allennlp.commands import Subcommand


class C(Subcommand):
    def add_subparser(
        self, name: str, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        raise NotImplementedError
