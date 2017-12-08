"""
Add arbitrary files to the model archive.
"""
from typing import Any, Iterable, Dict, Set
import inspect

class FileArchiver:
    """
    If a class inherits from this, some of the files it depends on
    in its ``from_params`` method will be included in the ``model.tar.gz`` archive.
    Then when reconstituted from the archive it will be supplied with paths
    to the archived files rather than the original files (which may not exist any longer).

    In order to make this work, you need to do two things:

    (1) in the ``from_params()`` method, you will need to create the instance
        and then assign

            instance._param_history = params.history

        before returning it. This is necessary so that when it's time to archive the model,
        we'll know where to modify the params.

    (2) you'll need to override the ``files_to_archive`` method to return a dict

            { key_in_params -> supplied_filename }

        that will indicate the key (as expected by the ``from_params`` method)
        and corresponding filename. The archiving process will then include this
        file in the ``model.tar.gz`` and likewise use it when restoring the archived model.
    """
    _param_history: str = None

    def files_to_archive(self) -> Dict[str, str]:  # pylint: disable=no-self-use
        """
        A class that implements ``Archivable`` should override this method which simply
        returns a ``dict`` {name -> filename} of files it wants archived.

        These files will get included in the `model.tar.gz`, and then during archiving
        the relevant params will get replaced with temporary paths to them.
        """
        return {}

def collect(obj: Any) -> Dict[str, str]:
    """
    Given some object which may be a ``FileArchiver`` and may have members
    that contain other ``FileArchiver`` s, collects all the files to archive
    in a dict { hocon_configuration_path -> filename }.
    """
    result = {}
    for file_archiver in _find_all_file_archivers(obj):
        param_history = file_archiver._param_history  # pylint: disable=protected-access
        for name, filename in file_archiver.files_to_archive().items():
            if param_history is None:
                raise RuntimeError("`FileArchiver` subclasses must set `_param_history` "
                                   "in their `from_params` method")
            result[f"{param_history}{name}"] = filename

    return result


def _find_all_file_archivers(obj: Any, seen: Set[int] = None) -> Iterable[FileArchiver]:
    """
    Given an object and a prefix, recursively explores and yields
    all ``Archivable`` members along with unique "paths" to them. It will explore
    list elements, dict values, and instances of classes.
    """
    # Track ids of explored objects to avoid infinite loops.
    this_id = id(obj)

    if seen and this_id in seen:
        # Already explored this object.
        return
    else:
        # Add this object to ``seen``
        seen = (seen or set()) | {this_id}

    # Start by yielding this instance
    if isinstance(obj, FileArchiver):
        yield obj

    # ``dir`` gets way too many properties, so filter intelligently.
    for name in dir(obj):
        # Retrieve the property with this name
        prop = getattr(obj, name)
        # Skip __ methods and the registry
        if name.startswith("__") or name == "_registry":
            continue
        # Skip functions and methods
        if inspect.isroutine(prop):
            continue
        # Skip simple types
        if prop.__class__ in ['str', 'int', 'float']:
            continue
        # Skip torch classes:
        if prop.__class__.__module__.split(".")[0] == "torch":
            continue
        # For a dictionary, explore all the values
        if isinstance(prop, dict):
            for subobj in prop.values():
                yield from _find_all_file_archivers(subobj, seen)
        # For a list, explore all the items
        elif isinstance(prop, list):
            for subobj in prop:
                yield from _find_all_file_archivers(subobj, seen)
        # For a class, recurse
        else:
            yield from _find_all_file_archivers(prop, seen)
