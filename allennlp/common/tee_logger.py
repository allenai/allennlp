"""
A logger that maintains logs of both stdout and stderr when models are run.
"""

from typing import TextIO
import os


class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::

        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """
    def __init__(self, filename: str, terminal: TextIO) -> None:
        self.terminal = terminal
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        # We'll special case a particular thing that TQDM does, to make the log file more
        # readable.  TQDM uses carriage returns to get the training line to update for each batch
        # without adding more lines to the terminal output.  Displaying those in a file won't work
        # correctly, so we'll just make sure that each batch shows up on its one line.
        if '\r' in message:
            message = message.replace('\r', '')
            if not message or message[-1] != '\n':
                message += '\n'
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
