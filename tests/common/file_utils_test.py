# pylint: disable=no-self-use,invalid-name
import os
import pytest

import responses

from allennlp.common.file_utils import url_to_filename, filename_to_url, get_from_cache, cached_path
from allennlp.common.testing import AllenNlpTestCase


class TestFileUtils(AllenNlpTestCase):

    def test_url_to_filename(self):
        for url in ['http://allenai.org', 'http://allennlp.org',
                    'https://www.google.com', 'http://pytorch.org']:
            filename = url_to_filename(url)
            assert "http" not in filename
            back_to_url = filename_to_url(filename)
            assert back_to_url == url

    @responses.activate
    def test_get_from_cache(self):
        url = 'http://fake.datastore.com/glove.txt.gz'

        with open('tests/fixtures/glove.6B.100d.sample.txt.gz', 'rb') as glove:
            glove_bytes = glove.read()

        # Mock response for the datastore url that returns glove vectors
        responses.add(
                responses.GET,
                url,
                body=glove_bytes,
                status=200,
                content_type='application/gzip',
                stream=True,
                headers={'Content-Length': str(len(glove_bytes))}
        )

        filename = get_from_cache(url, cache_dir=self.TEST_DIR)

        # We should have made a HTTP call and cached the result
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url))
        assert len(responses.calls) == 1

        # And the cached file should have the correct contents
        with open(filename, 'rb') as cached_file:
            assert cached_file.read() == glove_bytes

        # A second call to `get_from_cache` should not make a HTTP call
        # but should just return the cached filename.
        filename2 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename2 == filename
        assert len(responses.calls) == 1

        with open(filename2, 'rb') as cached_file:
            assert cached_file.read() == glove_bytes


    @responses.activate
    def test_cached_path(self):
        url = 'http://fake.datastore.com/glove.txt.gz'
        glove_file = 'tests/fixtures/glove.6B.100d.sample.txt.gz'

        with open(glove_file, 'rb') as glove:
            glove_bytes = glove.read()

        # Mock response for the datastore url that returns glove vectors
        responses.add(
                responses.GET,
                url,
                body=glove_bytes,
                status=200,
                content_type='application/gzip',
                stream=True,
                headers={'Content-Length': str(len(glove_bytes))}
        )

        # non-existent file
        with pytest.raises(FileNotFoundError):
            filename = cached_path("tests/fixtures/does_not_exist/fake_file.tar.gz")

        # unparsable URI
        with pytest.raises(ValueError):
            filename = cached_path("fakescheme://path/to/fake/file.tar.gz")

        # existing file as path
        assert cached_path(glove_file) == glove_file

        # caches urls
        filename = cached_path(url, cache_dir=self.TEST_DIR)

        assert len(responses.calls) == 1
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url))

        with open(filename, 'rb') as cached_file:
            assert cached_file.read() == glove_bytes
