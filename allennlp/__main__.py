#!/usr/bin/env python
import logging
import os
import sys
import warnings


if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("ALLENNLP_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

# filelock emits too many messages, so tell it to be quiet unless it has something
# important to say.
logging.getLogger("filelock").setLevel(logging.WARNING)


# transformers emits an annoying log message everytime it's imported, so we filter that
# one message out specifically.
def _transformers_log_filter(record):
    if record.msg.startswith("PyTorch version"):
        return False
    return True


logging.getLogger("transformers.file_utils").addFilter(_transformers_log_filter)


def run():
    # We issue a seperate warning from the tango command and ignore this one so that
    # users won't see a Tango warning when they're not using the Tango command.
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="AllenNLP Tango", module=r"allennlp\.tango"
    )

    from allennlp.commands import main  # noqa
    from allennlp.common.util import install_sigterm_handler

    # We want to be able to catch SIGTERM signals in addition to SIGINT (keyboard interrupt).
    install_sigterm_handler()

    main(prog="allennlp")


if __name__ == "__main__":
    run()
