#!/usr/bin/env python
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    level = logging.DEBUG
else:
    level = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=level)

from allennlp.commands import main  # pylint: disable=wrong-import-position

if __name__ == "__main__":
    main(prog="python -m allennlp.run")
