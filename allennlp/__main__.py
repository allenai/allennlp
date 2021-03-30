#!/usr/bin/env python
import logging
import os
import sys

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

from allennlp.commands import main  # noqa


def run():
    main(prog="allennlp")


if __name__ == "__main__":
    run()
