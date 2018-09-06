# pylint: disable=no-self-use,invalid-name
from collections import Counter
import os
import pathlib
import json
import tempfile
from typing import List, Tuple

import boto3
from moto import mock_s3
import pytest
import responses

from allennlp.common.file_utils import (
        url_to_filename, filename_to_url, get_from_cache, cached_path, split_s3_path,
        s3_request, s3_etag, s3_get)
from allennlp.common.testing import AllenNlpTestCase


def set_up_glove(url: str, byt: bytes, change_etag_every: int = 1000):
    # Mock response for the datastore url that returns glove vectors
    responses.add(
            responses.GET,
            url,
            body=byt,
            status=200,
            content_type='application/gzip',
            stream=True,
            headers={'Content-Length': str(len(byt))}
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

    responses.add_callback(
            responses.HEAD,
            url,
            callback=head_callback
    )


def set_up_s3_bucket(bucket_name: str = "my-bucket", s3_objects: List[Tuple[str, str]] = None):
    """Creates a mock s3 bucket optionally with objects uploaded from local files."""
    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=bucket_name)
    for filename, key in s3_objects or []:
        s3_client.upload_file(Filename=filename, Bucket=bucket_name, Key=key)


class TestFileUtils(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.glove_file = self.FIXTURES_ROOT / 'embeddings/glove.6B.100d.sample.txt.gz'
        with open(self.glove_file, 'rb') as glove:
            self.glove_bytes = glove.read()

    def test_url_to_filename(self):
        for url in ['http://allenai.org', 'http://allennlp.org',
                    'https://www.google.com', 'http://pytorch.org',
                    'https://s3-us-west-2.amazonaws.com/allennlp' + '/long' * 20 + '/url']:
            filename = url_to_filename(url)
            assert "http" not in filename
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            json.dump({'url': url, 'etag': None},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag is None

    def test_url_to_filename_with_etags(self):
        for url in ['http://allenai.org', 'http://allennlp.org',
                    'https://www.google.com', 'http://pytorch.org']:
            filename = url_to_filename(url, etag="mytag")
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({'url': url, 'etag': 'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == "mytag"
        baseurl = 'http://allenai.org/'
        assert url_to_filename(baseurl + '1') != url_to_filename(baseurl, etag='1')

    def test_url_to_filename_with_etags_eliminates_quotes(self):
        for url in ['http://allenai.org', 'http://allennlp.org',
                    'https://www.google.com', 'http://pytorch.org']:
            filename = url_to_filename(url, etag='"mytag"')
            assert "http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({'url': url, 'etag': 'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + '.json'), 'w'))
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

    @pytest.mark.skip(reason="moto mock library is broken: https://github.com/spulec/moto/issues/1793")
    @mock_s3
    def test_s3_bucket(self):
        """This just ensures the bucket gets set up correctly."""
        set_up_s3_bucket()
        s3_client = boto3.client("s3")
        buckets = s3_client.list_buckets()["Buckets"]
        assert len(buckets) == 1
        assert buckets[0]["Name"] == "my-bucket"

    @pytest.mark.skip(reason="moto mock library is broken: https://github.com/spulec/moto/issues/1793")
    @mock_s3
    def test_s3_request_wrapper(self):
        set_up_s3_bucket(s3_objects=[(str(self.glove_file), "embeddings/glove.txt.gz")])
        s3_resource = boto3.resource("s3")

        @s3_request
        def get_file_info(url):
            bucket_name, s3_path = split_s3_path(url)
            return s3_resource.Object(bucket_name, s3_path).content_type

        # Good request, should work.
        assert get_file_info("s3://my-bucket/embeddings/glove.txt.gz") == "text/plain"

        # File missing, should raise FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            get_file_info("s3://my-bucket/missing_file.txt")

    @pytest.mark.skip(reason="moto mock library is broken: https://github.com/spulec/moto/issues/1793")
    @mock_s3
    def test_s3_etag(self):
        set_up_s3_bucket(s3_objects=[(str(self.glove_file), "embeddings/glove.txt.gz")])
        # Ensure we can get the etag for an s3 object and that it looks as expected.
        etag = s3_etag("s3://my-bucket/embeddings/glove.txt.gz")
        assert isinstance(etag, str)
        assert etag.startswith("'") or etag.startswith('"')

        # Should raise FileNotFoundError if the file does not exist on the bucket.
        with pytest.raises(FileNotFoundError):
            s3_etag("s3://my-bucket/missing_file.txt")

    @pytest.mark.skip(reason="moto mock library is broken: https://github.com/spulec/moto/issues/1793")
    @mock_s3
    def test_s3_get(self):
        set_up_s3_bucket(s3_objects=[(str(self.glove_file), "embeddings/glove.txt.gz")])

        with tempfile.NamedTemporaryFile() as temp_file:
            s3_get("s3://my-bucket/embeddings/glove.txt.gz", temp_file)
            assert os.stat(temp_file.name).st_size != 0

        # Should raise FileNotFoundError if the file does not exist on the bucket.
        with pytest.raises(FileNotFoundError):
            with tempfile.NamedTemporaryFile() as temp_file:
                s3_get("s3://my-bucket/missing_file.txt", temp_file)

    @responses.activate
    def test_get_from_cache(self):
        url = 'http://fake.datastore.com/glove.txt.gz'
        set_up_glove(url, self.glove_bytes, change_etag_every=2)

        filename = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag="0"))

        # We should have made one HEAD request and one GET request.
        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts['HEAD'] == 1
        assert method_counts['GET'] == 1

        # And the cached file should have the correct contents
        with open(filename, 'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A second call to `get_from_cache` should make another HEAD call
        # but not another GET call.
        filename2 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename2 == filename

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts['HEAD'] == 2
        assert method_counts['GET'] == 1

        with open(filename2, 'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A third call should have a different ETag and should force a new download,
        # which means another HEAD call and another GET call.
        filename3 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename3 == os.path.join(self.TEST_DIR, url_to_filename(url, etag="1"))

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts['HEAD'] == 3
        assert method_counts['GET'] == 2

        with open(filename3, 'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

    @responses.activate
    def test_cached_path(self):
        url = 'http://fake.datastore.com/glove.txt.gz'
        set_up_glove(url, self.glove_bytes)

        # non-existent file
        with pytest.raises(FileNotFoundError):
            filename = cached_path(self.FIXTURES_ROOT / "does_not_exist" /
                                   "fake_file.tar.gz")

        # unparsable URI
        with pytest.raises(ValueError):
            filename = cached_path("fakescheme://path/to/fake/file.tar.gz")

        # existing file as path
        assert cached_path(self.glove_file) == str(self.glove_file)

        # caches urls
        filename = cached_path(url, cache_dir=self.TEST_DIR)

        assert len(responses.calls) == 2
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag="0"))

        with open(filename, 'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes
