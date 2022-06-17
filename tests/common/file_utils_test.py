import os
import pathlib
import json
import time
import shutil

from filelock import Timeout
import pytest
import responses
import torch

from allennlp.common.file_utils import (
    FileLock,
    _resource_to_filename,
    filename_to_url,
    cached_path,
    open_compressed,
    CacheFile,
    _Meta,
    _find_entries,
    inspect_cache,
    remove_cache_entries,
    LocalCacheResource,
    TensorCache,
)
from allennlp.common import Params
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.common.testing import AllenNlpTestCase
from allennlp.predictors import Predictor


def set_up_glove(url: str, byt: bytes, change_etag_every: int = 1000):
    # Mock response for the datastore url that returns glove vectors
    responses.add(
        responses.GET,
        url,
        body=byt,
        status=200,
        content_type="application/gzip",
        stream=True,
        headers={"Content-Length": str(len(byt))},
    )

    etags_left = change_etag_every
    etag = "0"

    def head_callback(_):
        """
        Writing this as a callback allows different responses to different HEAD requests.
        In our case, we're going to change the ETag header every `change_etag_every`
        requests, which will allow us to simulate having a new version of the file.
        """
        nonlocal etags_left, etag
        headers = {"ETag": etag}
        # countdown and change ETag
        etags_left -= 1
        if etags_left <= 0:
            etags_left = change_etag_every
            etag = str(int(etag) + 1)
        return (200, headers, "")

    responses.add_callback(responses.HEAD, url, callback=head_callback)


class TestFileLock(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()

        # Set up a regular lock and a read-only lock.
        open(self.TEST_DIR / "lock", "a").close()
        open(self.TEST_DIR / "read_only_lock", "a").close()
        os.chmod(self.TEST_DIR / "read_only_lock", 0o555)

        # Also set up a read-only directory.
        os.mkdir(self.TEST_DIR / "read_only_dir", 0o555)

    def test_locking(self):
        with FileLock(self.TEST_DIR / "lock"):
            # Trying to acquire the lock again should fail.
            with pytest.raises(Timeout):
                with FileLock(self.TEST_DIR / "lock", timeout=0.1):
                    pass

        # Trying to acquire a lock when lacking write permissions on the file should fail.
        with pytest.raises(PermissionError):
            with FileLock(self.TEST_DIR / "read_only_lock"):
                pass

        # But this should only issue a warning if we set the `read_only_ok` flag to `True`.
        with pytest.warns(UserWarning, match="Lacking permissions"):
            with FileLock(self.TEST_DIR / "read_only_lock", read_only_ok=True):
                pass

        # However this should always fail when we lack write permissions and the file lock
        # doesn't exist yet.
        with pytest.raises(PermissionError):
            with FileLock(self.TEST_DIR / "read_only_dir" / "lock", read_only_ok=True):
                pass


class TestFileUtils(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.glove_file = self.FIXTURES_ROOT / "embeddings/glove.6B.100d.sample.txt.gz"
        with open(self.glove_file, "rb") as glove:
            self.glove_bytes = glove.read()

    def test_resource_to_filename(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
            "https://allennlp.s3.amazonaws.com" + "/long" * 20 + "/url",
        ]:
            filename = _resource_to_filename(url)
            assert "http" not in filename
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            json.dump(
                {"url": url, "etag": None},
                open(os.path.join(self.TEST_DIR, filename + ".json"), "w"),
            )
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag is None

    def test_resource_to_filename_with_etags(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
        ]:
            filename = _resource_to_filename(url, etag="mytag")
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump(
                {"url": url, "etag": "mytag"},
                open(os.path.join(self.TEST_DIR, filename + ".json"), "w"),
            )
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"
        baseurl = "http://allenai.org/"
        assert _resource_to_filename(baseurl + "1") != _resource_to_filename(baseurl, etag="1")

    def test_resource_to_filename_with_etags_eliminates_quotes(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
        ]:
            filename = _resource_to_filename(url, etag='"mytag"')
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump(
                {"url": url, "etag": "mytag"},
                open(os.path.join(self.TEST_DIR, filename + ".json"), "w"),
            )
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"

    @responses.activate
    def test_cached_path(self):
        url = "http://fake.datastore.com/glove.txt.gz"
        set_up_glove(url, self.glove_bytes)

        # non-existent file
        with pytest.raises(FileNotFoundError):
            filename = cached_path(self.FIXTURES_ROOT / "does_not_exist" / "fake_file.tar.gz")

        # unparsable URI
        with pytest.raises(ValueError):
            filename = cached_path("fakescheme://path/to/fake/file.tar.gz")

        # existing file as path
        assert cached_path(self.glove_file) == str(self.glove_file)

        # caches urls
        # filename = cached_path(url, cache_dir=self.TEST_DIR)

        # assert len(responses.calls) == 2
        # assert filename == os.path.join(self.TEST_DIR, _resource_to_filename(url, etag="0"))

        # with open(filename, "rb") as cached_file:
        #     assert cached_file.read() == self.glove_bytes

        # archives
        filename = cached_path(
            self.FIXTURES_ROOT / "common" / "quote.tar.gz!quote.txt",
            extract_archive=True,
            cache_dir=self.TEST_DIR,
        )
        with open(filename, "r") as f:
            assert f.read().startswith("I mean, ")

    @responses.activate
    def test_cached_path_http_err_handling(self):
        url_404 = "http://fake.datastore.com/does-not-exist"
        byt = b"Does not exist"
        for method in (responses.GET, responses.HEAD):
            responses.add(
                method,
                url_404,
                body=byt,
                status=404,
                headers={"Content-Length": str(len(byt))},
            )

        with pytest.raises(FileNotFoundError):
            cached_path(url_404, cache_dir=self.TEST_DIR)

    def test_extract_with_external_symlink(self):
        dangerous_file = self.FIXTURES_ROOT / "common" / "external_symlink.tar.gz"
        with pytest.raises(ValueError):
            cached_path(dangerous_file, extract_archive=True)

    @pytest.mark.parametrize("suffix", ["bz2", "gz", "xz"])
    def test_open_compressed(self, suffix: str):
        uncompressed_file = self.FIXTURES_ROOT / "embeddings/fake_embeddings.5d.txt"
        with open_compressed(uncompressed_file) as f:
            uncompressed_lines = [line.strip() for line in f]

        compressed_file = f"{uncompressed_file}.{suffix}"
        with open_compressed(compressed_file) as f:
            compressed_lines = [line.strip() for line in f]
        assert compressed_lines == uncompressed_lines

    def test_meta_backwards_compatible(self):
        url = "http://fake.datastore.com/glove.txt.gz"
        etag = "some-fake-etag"
        filename = os.path.join(self.TEST_DIR, _resource_to_filename(url, etag))
        with open(filename, "wb") as f:
            f.write(self.glove_bytes)
        with open(filename + ".json", "w") as meta_file:
            json.dump({"url": url, "etag": etag}, meta_file)
        meta = _Meta.from_path(filename + ".json")
        assert meta.resource == url
        assert meta.etag == etag
        assert meta.creation_time is not None
        assert meta.size == len(self.glove_bytes)

    def create_cache_entry(self, url: str, etag: str, as_extraction_dir: bool = False):
        filename = os.path.join(self.TEST_DIR, _resource_to_filename(url, etag))
        cache_path = filename
        if as_extraction_dir:
            cache_path = filename + "-extracted"
            filename = filename + "-extracted/glove.txt"
            os.mkdir(cache_path)
        with open(filename, "wb") as f:
            f.write(self.glove_bytes)
        open(cache_path + ".lock", "a").close()
        meta = _Meta(
            resource=url,
            cached_path=cache_path,
            etag=etag,
            creation_time=time.time(),
            size=len(self.glove_bytes),
            extraction_dir=as_extraction_dir,
        )
        meta.to_file()

    def test_inspect(self, capsys):
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-1")
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-2")
        self.create_cache_entry(
            "http://fake.datastore.com/glove.txt.gz", "etag-3", as_extraction_dir=True
        )

        inspect_cache(cache_dir=self.TEST_DIR)

        captured = capsys.readouterr()
        assert "http://fake.datastore.com/glove.txt.gz" in captured.out
        assert "2 versions cached" in captured.out
        assert "1 version extracted" in captured.out

    def test_inspect_with_patterns(self, capsys):
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-1")
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-2")
        self.create_cache_entry("http://other.fake.datastore.com/glove.txt.gz", "etag-4")

        inspect_cache(cache_dir=self.TEST_DIR, patterns=["http://fake.*"])

        captured = capsys.readouterr()
        assert "http://fake.datastore.com/glove.txt.gz" in captured.out
        assert "2 versions" in captured.out
        assert "http://other.fake.datastore.com/glove.txt.gz" not in captured.out

    def test_remove_entries(self):
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-1")
        self.create_cache_entry("http://fake.datastore.com/glove.txt.gz", "etag-2")
        self.create_cache_entry(
            "http://fake.datastore.com/glove.txt.gz", "etag-3", as_extraction_dir=True
        )
        self.create_cache_entry("http://other.fake.datastore.com/glove.txt.gz", "etag-4")
        self.create_cache_entry(
            "http://other.fake.datastore.com/glove.txt.gz", "etag-5", as_extraction_dir=True
        )

        reclaimed_space = remove_cache_entries(["http://fake.*"], cache_dir=self.TEST_DIR)
        assert reclaimed_space == 3 * len(self.glove_bytes)

        size_left, entries_left = _find_entries(cache_dir=self.TEST_DIR)
        assert size_left == 2 * len(self.glove_bytes)
        assert len(entries_left) == 1
        entry_left = list(entries_left.values())[0]
        # one regular cache file and one extraction dir
        assert len(entry_left[0]) == 1
        assert len(entry_left[1]) == 1

        # Now remove everything.
        remove_cache_entries(["*"], cache_dir=self.TEST_DIR)
        assert len(os.listdir(self.TEST_DIR)) == 0


class TestCachedPathWithArchive(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.tar_file = self.TEST_DIR / "utf-8.tar.gz"
        shutil.copyfile(
            self.FIXTURES_ROOT / "utf-8_sample" / "archives" / "utf-8.tar.gz", self.tar_file
        )
        self.zip_file = self.TEST_DIR / "utf-8.zip"
        shutil.copyfile(
            self.FIXTURES_ROOT / "utf-8_sample" / "archives" / "utf-8.zip", self.zip_file
        )

    def check_extracted(self, extracted: str):
        assert os.path.isdir(extracted)
        assert pathlib.Path(extracted).parent == self.TEST_DIR
        assert os.path.exists(os.path.join(extracted, "dummy.txt"))
        assert os.path.exists(os.path.join(extracted, "folder/utf-8_sample.txt"))
        assert os.path.exists(extracted + ".json")

    def test_cached_path_extract_local_tar(self):
        extracted = cached_path(self.tar_file, cache_dir=self.TEST_DIR, extract_archive=True)
        self.check_extracted(extracted)

    def test_cached_path_extract_local_zip(self):
        extracted = cached_path(self.zip_file, cache_dir=self.TEST_DIR, extract_archive=True)
        self.check_extracted(extracted)

    @responses.activate
    @pytest.mark.skip(reason="until cached-path/rich versions are resolved")
    def test_cached_path_extract_remote_tar(self):
        url = "http://fake.datastore.com/utf-8.tar.gz"
        byt = open(self.tar_file, "rb").read()

        responses.add(
            responses.GET,
            url,
            body=byt,
            status=200,
            content_type="application/tar+gzip",
            stream=True,
            headers={"Content-Length": str(len(byt))},
        )
        responses.add(
            responses.HEAD,
            url,
            status=200,
            headers={"ETag": "fake-etag"},
        )

        extracted = cached_path(url, cache_dir=self.TEST_DIR, extract_archive=True)
        assert extracted.endswith("-extracted")
        self.check_extracted(extracted)

    @responses.activate
    @pytest.mark.skip(reason="until cached-path/rich versions are resolved")
    def test_cached_path_extract_remote_zip(self):
        url = "http://fake.datastore.com/utf-8.zip"
        byt = open(self.zip_file, "rb").read()

        responses.add(
            responses.GET,
            url,
            body=byt,
            status=200,
            content_type="application/zip",
            stream=True,
            headers={"Content-Length": str(len(byt))},
        )
        responses.add(
            responses.HEAD,
            url,
            status=200,
            headers={"ETag": "fake-etag"},
        )

        extracted = cached_path(url, cache_dir=self.TEST_DIR, extract_archive=True)
        assert extracted.endswith("-extracted")
        self.check_extracted(extracted)


class TestCacheFile(AllenNlpTestCase):
    def test_temp_file_removed_on_error(self):
        cache_filename = self.TEST_DIR / "cache_file"
        with pytest.raises(IOError, match="I made this up"):
            with CacheFile(cache_filename) as handle:
                raise IOError("I made this up")
        assert not os.path.exists(handle.name)
        assert not os.path.exists(cache_filename)


class TestLocalCacheResource(AllenNlpTestCase):
    def test_local_cache_resource(self):
        with LocalCacheResource("some-computation", "version-1", cache_dir=self.TEST_DIR) as cache:
            assert not cache.cached()

            with cache.writer() as w:
                json.dump({"a": 1}, w)

        with LocalCacheResource("some-computation", "version-1", cache_dir=self.TEST_DIR) as cache:
            assert cache.cached()

            with cache.reader() as r:
                data = json.load(r)

            assert data["a"] == 1


class TestTensorCache(AllenNlpTestCase):
    def test_tensor_cache(self):
        cache = TensorCache(self.TEST_DIR / "cache")
        assert not cache.read_only

        # Insert some stuff into the cache.
        cache["a"] = torch.tensor([1, 2, 3])

        # Close cache.
        del cache

        # Now let's open another one in read-only mode.
        cache = TensorCache(self.TEST_DIR / "cache", read_only=True)
        assert cache.read_only

        # If we try to write we should get a ValueError
        with pytest.raises(ValueError, match="cannot write"):
            cache["b"] = torch.tensor([1, 2, 3])

        # But we should be able to retrieve from the cache.
        assert cache["a"].shape == (3,)

        # Close this one.
        del cache

        # Now we're going to tell the OS to make the cache file read-only.
        os.chmod(self.TEST_DIR / "cache", 0o444)
        os.chmod(self.TEST_DIR / "cache-lock", 0o444)

        # This time when we open the cache, it should automatically be set to read-only.
        with pytest.warns(UserWarning, match="cache will be read-only"):
            cache = TensorCache(self.TEST_DIR / "cache")
            assert cache.read_only

    def test_tensor_cache_open_twice(self):
        cache1 = TensorCache(self.TEST_DIR / "multicache")
        cache1["foo"] = torch.tensor([1, 2, 3])
        cache2 = TensorCache(self.TEST_DIR / "multicache")
        assert cache1 is cache2

    def test_tensor_cache_upgrade(self):
        cache0 = TensorCache(self.TEST_DIR / "upcache")
        cache0["foo"] = torch.tensor([1, 2, 3])
        del cache0

        cache1 = TensorCache(self.TEST_DIR / "upcache", read_only=True)
        cache2 = TensorCache(self.TEST_DIR / "upcache")
        assert not cache1.read_only
        assert not cache2.read_only
        assert torch.allclose(cache1["foo"], torch.tensor([1, 2, 3]))

        cache2["bar"] = torch.tensor([2, 3, 4])
        assert torch.allclose(cache1["bar"], cache2["bar"])


class TestHFHubDownload(AllenNlpTestCase):
    def test_cached_download(self):
        params = Params(
            {
                "options_file": "hf://lysandre/test-elmo-tiny/options.json",
                "weight_file": "hf://lysandre/test-elmo-tiny/lm_weights.hdf5",
            }
        )
        embedding_layer = ElmoTokenEmbedder.from_params(vocab=None, params=params)

        assert isinstance(
            embedding_layer, ElmoTokenEmbedder
        ), "Embedding layer badly instantiated from HF Hub."
        assert (
            embedding_layer.get_output_dim() == 32
        ), "Embedding layer badly instantiated from HF Hub."

    def test_snapshot_download(self):
        predictor = Predictor.from_path("hf://lysandre/test-simple-tagger-tiny")
        assert predictor._dataset_reader._token_indexers["tokens"].namespace == "test_tokens"

    def test_cached_download_no_user_or_org(self):
        path = cached_path("hf://t5-small/config.json", cache_dir=self.TEST_DIR)
        assert os.path.isfile(path)
        assert pathlib.Path(os.path.dirname(path)) == self.TEST_DIR
        assert os.path.isfile(path + ".json")
        meta = _Meta.from_path(path + ".json")
        assert meta.etag is not None
        assert meta.resource == "hf://t5-small/config.json"

    def test_snapshot_download_no_user_or_org(self):
        # This is the smallest snapshot I could find that is not associated with a user / org.
        model_name = "distilbert-base-german-cased"
        path = cached_path(f"hf://{model_name}")
        assert os.path.isdir(path)
        assert os.path.isfile(path + ".json")
        meta = _Meta.from_path(path + ".json")
        assert meta.resource == f"hf://{model_name}"
