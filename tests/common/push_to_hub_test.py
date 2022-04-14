import shutil
import time

import pytest
import requests
from huggingface_hub import HfApi, Repository
from unittest.mock import patch

from allennlp.common.push_to_hf import push_to_hf
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models import load_archive

ENDPOINT_STAGING = "https://moon-staging.huggingface.co"

USER = "__DUMMY_TRANSFORMERS_USER__"
PASS = "__DUMMY_TRANSFORMERS_PASS__"
ORG_NAME = "valid_org"


REPO_NAME = "my-allennlp-model-{}".format(int(time.time() * 10e3))


def with_staging_testing(func):
    file_download = patch(
        "huggingface_hub.file_download.HUGGINGFACE_CO_URL_TEMPLATE",
        ENDPOINT_STAGING + "/{repo_id}/resolve/{revision}/{filename}",
    )

    hf_api = patch(
        "huggingface_hub.constants.ENDPOINT",
        ENDPOINT_STAGING,
    )

    repository = patch(
        "huggingface_hub.constants.ENDPOINT",
        ENDPOINT_STAGING,
    )

    return repository(hf_api(file_download(func)))


@pytest.mark.skip(reason="This test does not work anymore with the new version of huggingface-hub.")
class TestPushToHub(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.api = HfApi(ENDPOINT_STAGING)
        self.token = self.api.login(username=USER, password=PASS)
        self.local_repo_path = self.TEST_DIR / "hub"
        self.clone_path = self.TEST_DIR / "hub_clone"

    def teardown_method(self):
        super().teardown_method()
        try:
            self.api.delete_repo(token=self.token, name=REPO_NAME)
        except requests.exceptions.HTTPError:
            pass

        try:
            self.api.delete_repo(
                token=self.token,
                organization=ORG_NAME,
                name=REPO_NAME,
            )
        except requests.exceptions.HTTPError:
            pass

    @with_staging_testing
    def test_push_to_hub_archive_path(self):
        archive_path = self.FIXTURES_ROOT / "simple_tagger" / "serialization_full" / "model.tar.gz"
        url = push_to_hf(
            repo_name=REPO_NAME,
            archive_path=archive_path,
            local_repo_path=self.local_repo_path,
            use_auth_token=self.token,
        )

        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

        Repository(
            self.clone_path,
            clone_from=f"{ENDPOINT_STAGING}/{USER}/{REPO_NAME}",
            use_auth_token=self.token,
        )
        load_archive(self.clone_path)
        shutil.rmtree(self.clone_path)

    @with_staging_testing
    def test_push_to_hub_serialization_dir(self):
        serialization_dir = self.FIXTURES_ROOT / "simple_tagger" / "serialization_full"
        url = push_to_hf(
            repo_name=REPO_NAME,
            serialization_dir=serialization_dir,
            local_repo_path=self.local_repo_path,
            use_auth_token=self.token,
        )

        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

        Repository(
            self.clone_path,
            clone_from=f"{ENDPOINT_STAGING}/{USER}/{REPO_NAME}",
            use_auth_token=self.token,
        )
        load_archive(self.clone_path)
        shutil.rmtree(self.clone_path)

    @with_staging_testing
    def test_push_to_hub_to_org(self):
        serialization_dir = self.FIXTURES_ROOT / "simple_tagger" / "serialization_full"
        url = push_to_hf(
            repo_name=REPO_NAME,
            serialization_dir=serialization_dir,
            organization=ORG_NAME,
            local_repo_path=self.local_repo_path,
            use_auth_token=self.token,
        )

        # Check that the returned commit url
        # actually exists.
        r = requests.head(url)
        r.raise_for_status()

        Repository(
            self.clone_path,
            clone_from=f"{ENDPOINT_STAGING}/{ORG_NAME}/{REPO_NAME}",
            use_auth_token=self.token,
        )
        load_archive(self.clone_path)
        shutil.rmtree(self.clone_path)

    @with_staging_testing
    def test_push_to_hub_fails_with_invalid_token(self):
        serialization_dir = self.FIXTURES_ROOT / "simple_tagger" / "serialization_full"
        with pytest.raises(ValueError):
            push_to_hf(
                repo_name=REPO_NAME,
                serialization_dir=serialization_dir,
                local_repo_path=self.local_repo_path,
                use_auth_token="invalid token",
            )
