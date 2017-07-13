from collections import OrderedDict
from typing import Dict, Type  # pylint: disable=unused-import

from allennlp.experiments.driver import Driver
from allennlp.experiments.train_driver import TrainDriver


# pylint: disable=invalid-name
drivers = OrderedDict()  # type: Dict[str, Type[Driver]]
# pylint: enable=invalid-name

drivers['train'] = TrainDriver
