from os import PathLike
from dataclasses import dataclass, asdict
import json
import logging
import socket
from git import Repo
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
            json.dump(asdict(self).update(get_git_info()), meta_file)

    @classmethod
    def from_path(cls, path: Union[PathLike, str]) -> "Meta":
        with open(path) as meta_file:
            data = json.load(meta_file)
        return cls(**data)


def get_git_info():
    repo = Repo(search_parent_directories=True)
    repo_info = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "hostname": str(socket.gethostname()),
    }
    return repo_info
