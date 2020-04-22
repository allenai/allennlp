from collections import Counter
import os
import pathlib
import json

import pytest
import responses

from allennlp.common.file_utils import (
    url_to_filename,
    filename_to_url,
    get_from_cache,
    cached_path,
    split_s3_path,
    open_compressed,
)
from allennlp.common.testing import AllenNlpTestCase


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


class TestFileUtils(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.glove_file = self.FIXTURES_ROOT / "embeddings/glove.6B.100d.sample.txt.gz"
        with open(self.glove_file, "rb") as glove:
            self.glove_bytes = glove.read()

    def test_url_to_filename(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
            "https://allennlp.s3.amazonaws.com" + "/long" * 20 + "/url",
        ]:
            filename = url_to_filename(url)
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

    def test_url_to_filename_with_etags(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
        ]:
            filename = url_to_filename(url, etag="mytag")
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
        assert url_to_filename(baseurl + "1") != url_to_filename(baseurl, etag="1")

    def test_url_to_filename_with_etags_eliminates_quotes(self):
        for url in [
            "http://allenai.org",
            "http://allennlp.org",
            "https://www.google.com",
            "http://pytorch.org",
        ]:
            filename = url_to_filename(url, etag='"mytag"')
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump(
                {"url": url, "etag": "mytag"},
                open(os.path.join(self.TEST_DIR, filename + ".json"), "w"),
            )
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"

    def test_split_s3_path(self):
        # Test splitting good urls.
        assert split_s3_path("s3://my-bucket/subdir/file.txt") == ("my-bucket", "subdir/file.txt")
        assert split_s3_path("s3://my-bucket/file.txt") == ("my-bucket", "file.txt")

        # Test splitting bad urls.
        with pytest.raises(ValueError):
            split_s3_path("s3://")
            split_s3_path("s3://myfile.txt")
            split_s3_path("myfile.txt")

    @responses.activate
    def test_get_from_cache(self):
        url = "http://fake.datastore.com/glove.txt.gz"
        set_up_glove(url, self.glove_bytes, change_etag_every=2)

        filename = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag="0"))

        # We should have made one HEAD request and one GET request.
        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts["HEAD"] == 1
        assert method_counts["GET"] == 1

        # And the cached file should have the correct contents
        with open(filename, "rb") as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A second call to `get_from_cache` should make another HEAD call
        # but not another GET call.
        filename2 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename2 == filename

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts["HEAD"] == 2
        assert method_counts["GET"] == 1

        with open(filename2, "rb") as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A third call should have a different ETag and should force a new download,
        # which means another HEAD call and another GET call.
        filename3 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename3 == os.path.join(self.TEST_DIR, url_to_filename(url, etag="1"))

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts["HEAD"] == 3
        assert method_counts["GET"] == 2

        with open(filename3, "rb") as cached_file:
            assert cached_file.read() == self.glove_bytes

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
        filename = cached_path(url, cache_dir=self.TEST_DIR)

        assert len(responses.calls) == 2
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag="0"))

        with open(filename, "rb") as cached_file:
            assert cached_file.read() == self.glove_bytes

    def test_open_compressed(self):
        uncompressed_file = self.FIXTURES_ROOT / "embeddings/fake_embeddings.5d.txt"
        with open_compressed(uncompressed_file) as f:
            uncompressed_lines = [line.strip() for line in f]

        for suffix in ["bz2", "gz"]:
            compressed_file = f"{uncompressed_file}.{suffix}"
            with open_compressed(compressed_file) as f:
                compressed_lines = [line.strip() for line in f]
            assert compressed_lines == uncompressed_lines
