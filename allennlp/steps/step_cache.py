import json
import logging
import pathlib
import weakref
from os import PathLike
from typing import MutableMapping, Any, Dict, Union

from allennlp.common import Registrable
from allennlp.steps.step import Step

logger = logging.getLogger(__name__)


class StepCache(MutableMapping[Step, Any], Registrable):
    def __delitem__(self, key: Step):
        raise ValueError("Cached results are forever.")

    def __iter__(self):
        raise ValueError("Step caches are not iterable.")

    def __contains__(self, step: Step) -> bool:
        """This is a generic implementation of __contains__. If you are writing your own
        `StepCache`, you might want to write a faster one yourself."""
        try:
            self.__getitem__(step)
            return True
        except KeyError:
            return False

    def __getitem__(self, step: Step) -> Any:
        raise NotImplementedError()

    def __setitem__(self, step: Step, value: Any) -> None:
        raise NotImplementedError()


@StepCache.register("memory")
class MemoryStepCache(StepCache):
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def __getitem__(self, step: Step) -> Any:
        return self.cache[step.unique_id()]

    def __setitem__(self, step: Step, value: Any) -> None:
        if step.cache_results:
            self.cache[step.unique_id()] = value
        else:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)

    def __contains__(self, step: Step):
        return step.unique_id() in self.cache

    def __len__(self) -> int:
        return len(self.cache)


default_step_cache = MemoryStepCache()


@StepCache.register("directory")
class DirectoryStepCache(StepCache):
    def __init__(self, dir: Union[str, PathLike]):
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        # We keep an in-memory cache as well so we don't have to de-serialize stuff
        # we happen to have in memory already.
        self.cache = weakref.WeakValueDictionary()

    def __contains__(self, step: Step):
        if step.unique_id() in self.cache:
            return True
        return self.path_for_step(step).exists()

    def __getitem__(self, step: Step) -> Any:
        try:
            return self.cache[step.unique_id()]
        except KeyError:
            result = step.format.read(self.path_for_step(step))
            self.cache[step.unique_id()] = result
            return result

    def __setitem__(self, step: Step, value: Any) -> None:
        location = self.path_for_step(step)
        step.format.write(value, location)
        metadata = {
            "step": step.unique_id(),
            "checksum": step.format.checksum(location),
        }
        with (location / "metadata.json").open("wt") as f:
            json.dump(metadata, f)
        self.cache[step.unique_id()] = value

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*/metadata.json"))

    def path_for_step(self, step: Step) -> pathlib.Path:
        return self.dir / step.unique_id()
