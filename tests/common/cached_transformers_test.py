from shutil import rmtree
import pytest
from tempfile import mkdtemp

from allennlp.common import cached_transformers

def test_get_missing_from_cache_local_files_only():
    tempdir = mkdtemp()
    try:
        with pytest.raises(ValueError) as execinfo:
            cached_transformers.get(
                "bert-base-uncased",
                True,
                transformers_from_pretrained_kwargs={
                    "cache_dir": tempdir,
                    "local_files_only": True,
                },
            )
        assert str(execinfo.value) == (
            "Cannot find the requested files in the cached path and "
            "outgoing traffic has been disabled. To enable model "
            "look-ups and downloads online, set 'local_files_only' "
            "to False."
        )
    finally:
        rmtree(tempdir)


def test_get_tokenizer_missing_from_cache_local_files_only():
    tempdir = mkdtemp()
    try:
        with pytest.raises(ValueError) as execinfo:
            cached_transformers.get_tokenizer(
                "bert-base-uncased",
                cache_dir=tempdir,
                local_files_only=True,
            )
        assert str(execinfo.value) == (
            "Cannot find the requested files in the cached path and "
            "outgoing traffic has been disabled. To enable model "
            "look-ups and downloads online, set 'local_files_only' "
            "to False."
        )
    finally:
        rmtree(tempdir)
