from typing import TextIO
import os


def replace_cr_with_newline(message: str) -> str:
    """
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output. Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    """
    if "\r" in message:
        message = message.replace("\r", "")
        if not message or message[-1] != "\n":
            message += "\n"
    return message


class TeeHandler:
    """
    This class behaves similar to the `tee` command-line utility by writing messages
    to both a terminal and a file.

    It's meant to be used like this to patch sys.stdout and sys.stderr.

    # Examples

    ```python
    sys.stdout = TeeHandler("stdout.log", sys.stdout)
    sys.stderr = TeeHandler("stdout.log", sys.stderr)
    ```
    """

    def __init__(
        self,
        filename: str,
        terminal: TextIO,
        file_friendly_terminal_output: bool = False,
        silent: bool = False,
    ) -> None:
        self.terminal = terminal
        self.file_friendly_terminal_output = file_friendly_terminal_output
        self.silent = silent
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        cleaned = replace_cr_with_newline(message)

        if not self.silent:
            if self.file_friendly_terminal_output:
                self.terminal.write(cleaned)
            else:
                self.terminal.write(message)

        self.log.write(cleaned)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        # Mirror the API of sys.stdout so that we can
        # check for the presence of a terminal easily.
        return not self.file_friendly_terminal_output

    def cleanup(self) -> TextIO:
        self.log.close()
        return self.terminal
