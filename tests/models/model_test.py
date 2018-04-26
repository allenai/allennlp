# pylint: disable=no-self-use
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

@Model.register('not-overridden')
class NotOverridden(Model):
    pass

@Model.register('yes-overridden')
class YesOverridden(Model):
    def __init__(self, vocab) -> None:
        super().__init__(vocab)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'YesOverridden':
        return YesOverridden(vocab)

class TestModel:
    def test_raises_without_override(self):
        params = Params({'type': 'not-overridden'})

        with pytest.raises(ConfigurationError) as exc:
            _ = Model.from_params(None, params)

        assert exc.value.message == "You must override Model.from_params in your subclass"

    def test_succeeds_with_override(self):
        params = Params({'type': 'yes-overridden'})
        _ = Model.from_params(None, params)
