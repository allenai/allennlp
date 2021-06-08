from os import PathLike
from dataclasses import dataclass, asdict
import json
import logging
from typing import Union

from allennlp.version import VERSION


logger = logging.getLogger(__name__)


META_NAME = "meta.json"


@dataclass
class Meta:
    """
    Defines the meta data that's saved in a serialization directory and archive
    when training an AllenNLP model.
    """

    version: str

    @classmethod
    def new(cls) -> "Meta":
        return cls(version=VERSION)

    def to_file(self, path: Union[PathLike, str]) -> None:
        with open(path, "w") as meta_file:
            json.dump(asdict(self), meta_file)

    @classmethod
    def from_path(cls, path: Union[PathLike, str]) -> "Meta":
        with open(path) as meta_file:
            data = json.load(meta_file)
        return cls(**data)
