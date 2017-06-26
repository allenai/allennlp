import io
import os


class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::

        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """
    def __init__(self, filename: str, terminal: io.TextIOWrapper):
        self.terminal = terminal
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        # We'll special case a particular thing that keras does, to make the log file more
        # readable.  Keras uses ^H characters to get the training line to update for each batch
        # without adding more lines to the terminal output.  Displaying those in a file won't work
        # correctly, so we'll just make sure that each batch shows up on its one line.
        if '\x08' in message:
            message = message.replace('\x08', '')
            if len(message) == 0 or message[-1] != '\n':
                message += '\n'
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
