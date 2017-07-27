import logging
import os
import sys

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from allennlp.run import train_model_from_file
from allennlp.common.checks import ensure_pythonhashseed_set

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    usage = 'USAGE: train.py [param_file] [train]'
    if len(sys.argv) == 3:

        if sys.argv[2] == "train":
            train_model_from_file(sys.argv[1])
        else:
            raise NotImplementedError
    else:
        print(usage)
        sys.exit(-1)

if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
