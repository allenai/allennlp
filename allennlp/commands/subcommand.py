"""
Base class for subcommands under ``allennlp.run``.
"""
import argparse

class Subcommand:
    """
    An abstract class representing subcommands for allennlp.run.
    If you wanted to (for example) create your own custom `special-evaluate` command to use like

    ``allennlp special-evaluate ...``

    you would create a ``Subcommand`` subclass and then pass it as an override to
    :func:`~allennlp.commands.main` .
    """
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        raise NotImplementedError
