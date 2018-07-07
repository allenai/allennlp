# pylint: disable=no-self-use,invalid-name
import pytest

from typing import cast

from allennlp.data.dataset_readers import Event2MindDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

def get_text(instance: Instance, key: str):
    return [t.text for t in cast(TextField, instance.fields[key]).tokens]

def get_source(instance: Instance):
    return get_text(instance, "source_tokens")

def get_target(instance: Instance):
    return get_text(instance, "target_tokens")

class TestEvent2MindDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = Event2MindDatasetReader(lazy=lazy)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'event2mind.csv'))
        instances = ensure_list(instances)

        assert len(instances) == 4
        instance = instances[0]
        assert get_source(instance) == ["@start@", "It", "is", "PersonX", "'s",
                                        "favorite", "animal", "@end@"]
        assert get_target(instance) == ["@start@", "none", "@end@"]
        instance = instances[1]
        assert get_source(instance) == ["@start@", "PersonX", "drives",
                                        "PersonY", "'s", "truck", "@end@"]
        assert get_target(instance) == ["@start@", "to", "move", "@end@"]
        instance = instances[2]
        assert get_source(instance) == ["@start@", "PersonX", "drives",
                                        "PersonY", "'s", "truck", "@end@"]
        assert get_target(instance) == ["@start@", "to", "steal", "@end@"]
        instance = instances[3]
        assert get_source(instance) == ["@start@", "PersonX", "gets", "PersonY",
                                        "'s", "mother", "@end@"]
        assert get_target(instance) == ["@start@", "to", "be", "helpful", "@end@"]
