import collections
import hashlib
import io
from typing import Any, MutableMapping

import base58
import dill


class CustomDetHash:
    def det_hash_object(self) -> Any:
        """
        By default, `det_hash()` pickles an object, and returns the hash of the pickled
        representation. Sometimes you want to take control over what goes into
        that hash. In that case, implement this method. `det_hash()` will pickle the
        result of this method instead of the object itself.

        If you return `None`, `det_hash()` falls back to the original behavior and pickles
        the object.
        """
        raise NotImplementedError()


class DetHashFromInitParams(CustomDetHash):
    """
    Add this class as a mixin base class to make sure your class's det_hash is derived
    exclusively from the parameters passed to __init__().
    """

    _det_hash_object: Any

    def __new__(cls, *args, **kwargs):
        super_new = super(DetHashFromInitParams, cls).__new__
        if super().__new__ is object.__new__ and cls.__init__ is not object.__init__:
            instance = super_new(cls)
        else:
            instance = super_new(cls, *args, **kwargs)
        instance._det_hash_object = (args, kwargs)
        return instance

    def det_hash_object(self) -> Any:
        return self._det_hash_object


class DetHashWithVersion(CustomDetHash):
    """
    Add this class as a mixing base class to make sure your class's det_hash can be modified
    by altering a static `VERSION` member of your class.
    """

    VERSION = None

    def det_hash_object(self) -> Any:
        if self.VERSION is not None:
            return self.VERSION, self
        else:
            return None


class _DetHashPickler(dill.Pickler):
    def __init__(self, buffer: io.BytesIO):
        super().__init__(buffer)

        # We keep track of how deeply we are nesting the pickling of an object.
        # If a class returns `self` as part of `det_hash_object()`, it causes an
        # infinite recursion, because we try to pickle the `det_hash_object()`, which
        # contains `self`, which returns a `det_hash_object()`, etc.
        # So we keep track of how many times recursively we are trying to pickle the
        # same object. We only call `det_hash_object()` the first time. We assume that
        # if `det_hash_object()` returns `self` in any way, we want the second time
        # to just pickle the object as normal. `DetHashWithVersion` takes advantage
        # of this ability.
        self.recursively_pickled_ids: MutableMapping[int, int] = collections.Counter()

    def save(self, obj, save_persistent_id=True):
        self.recursively_pickled_ids[id(obj)] += 1
        super().save(obj, save_persistent_id)
        self.recursively_pickled_ids[id(obj)] -= 1

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, CustomDetHash) and self.recursively_pickled_ids[id(obj)] <= 1:
            det_hash_object = obj.det_hash_object()
            if det_hash_object is not None:
                return obj.__class__.__module__, obj.__class__.__qualname__, det_hash_object
            else:
                return None
        elif isinstance(obj, type):
            return obj.__module__, obj.__qualname__
        else:
            return None


def det_hash(o: Any) -> str:
    """
    Returns a deterministic hash code of arbitrary Python objects.

    If you want to override how we calculate the deterministic hash, derive from the
    `CustomDetHash` class and implement `det_hash_object()`.
    """
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        pickler = _DetHashPickler(buffer)
        pickler.dump(o)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()
