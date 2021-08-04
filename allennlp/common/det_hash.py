import hashlib
import io
from typing import Any

import base58
import dill


class CustomDetHash:
    def det_hash_object(self) -> Any:
        """
        By default, `det_hash()` pickles an object, and returns the hash of the pickled
        representation. Sometimes you want to take control over what goes into
        that hash. In that case, implement this method. `det_hash()` will pickle the
        result of this method instead of the object itself.
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


class _DetHashPickler(dill.Pickler):
    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, CustomDetHash):
            return obj.__class__.__qualname__, obj.det_hash_object()
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
