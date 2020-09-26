import pytest

from allennlp.common.testing import AllenNlpTestCase


@pytest.mark.skip("makes test-install fail (and also takes 30 seconds)")
class TestBasicAllenNlp(AllenNlpTestCase):
    @classmethod
    def test_run_as_script(cls):
        # Just ensure the tutorial runs without throwing an exception.

        import tutorials.tagger.basic_allennlp  # noqa
