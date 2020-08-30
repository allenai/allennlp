# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import os
import shutil
import struct
import numpy as np
import torch
from allennlp.data.fields import DataArray

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return f"{prefix_path}.idx"


def data_file_path(prefix_path):
    return f"{prefix_path}.bin"


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapCacheReader:
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        # self._index.dtype will be different everytime
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)
        #To tensor_dict() here.
        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class MMapCacheBuilder(object):
    def __init__(self, out_file):
        self._data_file = open(out_file, "wb")
        self._sizes = []
        self._field_names = None
        self._dtype = np.int32

    def add_instance(self, instance: Instance):
        tensor_dict = instance.as_tensor_dict()
        flattened_dict = self.flatten_dict(tensor_dict)
        if not self._field_names:
            self._field_names = list(sorted(flattened_dict.keys()))
            assert self._field_names
            # TODO: what if some instances have a different set of field names, i.e missing some, for test instances, we don't have supervision.....
            # for now we will just write the name of every field next_field_names
            self.add_tensor(key, value)

    @classmethod
    def flatten_dict(cls, tensor_dict: Dict, prefix=None):
        flat_dict = {}
        for field_name, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                name = f"{prefix}___{field_name}" if prefix else field_name
                flat_dict[name] = value
            elif isinstance(value, dict):
                flat_dict.update(cls.flatten_dict(value, prefix=field_name))
            else:
                raise ValueError("You gave me a MetadataField")
        return flat_dict

    def add_tensor(self, name, tensor):
        np_array = tensor.contiguous().detach().numpy()
        np_array_b = np_array.tobytes(order="C")
        name_b = name.encode()
        self._sizes += [len(name_b), len(np_array_b), np_array.size]
        self._data_file.write(name_b)
        self._data_file.write(np_array_b)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapCacheReader.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapCacheReader.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)



class MMapCache:
    def __init__(
        self,
        cache_path: str,
    ) -> None:
        self.cache_path = cache_path
        self._builder = None
        self._cache = None
        
        if os.path.exists(self.cache_path)):
            if self.is_finalized(self.cache_path):
                #scenario 2, we can read.
                self._cache = MMapCacheReader(self.cache_path)
            else:
                #scenario 3, another training process is currently writing to it or was interrupted while it was writing.
                pass
        else:
            self._builder = MMapCacheBuilder(self.cache_path)
            #scenario 1, we need to write to it.
            
    def get_instances(
        self,
        data_path: str,
    ) -> Optional[Iterable[Dict[str, DataArray]]]:
        #dont need data_path here
        assert self._cache
        for i in range(len(self._cache)):
            yield self._cache[i]
        

    def set_instances(
        self,
        instances: Iterable[Dict[str, DataArray]],
    ) -> Iterable[Dict[str, DataArray]]:
        assert self._builder:
        for instance in instances:
            self._builder.add_instance(instance)
        return instances

            

    def get_vocabulary(self) -> Optional[Vocabulary]:
        pass

    def set_vocabulary(self, vocab: Vocabulary) -> None:
        pass

    def finalize(self) -> None:
        pass

    @classmethod
    def hash_config(cls, config: Params) -> str:
        pass
    
    @classmethod
    def is_finalized(cls,path):
        return True

    

    # Similar to the DatasetReader class, the Cache class will also have
    # getters and setters for WorkerInfo and DistributedInfo.