# pylint: disable=no-self-use,abstract-method
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader

@DatasetReader.register('not-overridden')
class NotOverridden(DatasetReader):
    pass

@DatasetReader.register('yes-overridden')
class YesOverridden(DatasetReader):
    @classmethod
    def from_params(cls, params: Params) -> 'YesOverridden':
        return YesOverridden()

class TestDatasetReader:
    def test_raises_without_override(self):
        params = Params({'type': 'not-overridden'})

        with pytest.raises(ConfigurationError) as exc:
            _ = DatasetReader.from_params(params)

        assert exc.value.message == "You must override DatasetReader.from_params in your subclass"

    def test_succeeds_with_override(self):
        params = Params({'type': 'yes-overridden'})
        _ = DatasetReader.from_params(params)
