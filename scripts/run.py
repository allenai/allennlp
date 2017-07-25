import logging
import os
import sys

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from allennlp.execute_driver import execute_driver_from_file
from allennlp.common.checks import ensure_pythonhashseed_set

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    usage = 'USAGE: run.py [param_file] [optional driver override]'
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        execute_driver_from_file(*sys.argv[1:])
    else:
        print(usage)
        sys.exit(-1)

if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
