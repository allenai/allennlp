# pylint: disable=no-self-use
# pylint: disable=invalid-name
from allennlp.data.semparse.worlds import NlvrWorld
from allennlp.data.dataset_readers import NlvrDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestNlvrDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        test_file = "tests/fixtures/data/nlvr/sample_data.jsonl"
        dataset = NlvrDatasetReader(add_paths_to_agenda=False).read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        assert instance.fields.keys() == {'sentence', 'agenda', 'world', 'actions', 'label'}
        sentence_tokens = instance.fields["sentence"].tokens
        expected_tokens = ['There', 'is', 'a', 'circle', 'closely', 'touching', 'a', 'corner', 'of',
                           'a', 'box', '.']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields["actions"].field_list]
        assert len(actions) == 121
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        agenda_strings = [actions[rule_id] for rule_id in agenda]
        assert set(agenda_strings) == set(['<o,o> -> circle',
                                           '<o,t> -> object_exists',
                                           '<o,o> -> touch_corner'])
        world = instance.fields["world"].as_tensor({})
        assert isinstance(world, NlvrWorld)
        label = instance.fields["label"].label
        assert label == "true"

    def test_agenda_indices_are_correct_without_paths(self):
        reader = NlvrDatasetReader(add_paths_to_agenda=False)
        test_file = "tests/fixtures/data/nlvr/sample_data.jsonl"
        dataset = reader.read(test_file)
        instances = list(dataset)
        instance = instances[0]
        sentence_tokens = instance.fields["sentence"].tokens
        sentence = " ".join([t.text for t in sentence_tokens])
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        actions = [action.rule for action in instance.fields["actions"].field_list]
        agenda_actions = [actions[i] for i in agenda]
        world = instance.fields["world"].as_tensor({})
        expected_agenda_actions = world.get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
        assert expected_agenda_actions == agenda_actions

    def test_agenda_indices_are_correct_with_paths(self):
        reader = NlvrDatasetReader()
        test_file = "tests/fixtures/data/nlvr/sample_data.jsonl"
        dataset = reader.read(test_file)
        instances = list(dataset)
        instance = instances[0]
        sentence_tokens = instance.fields["sentence"].tokens
        sentence = " ".join([t.text for t in sentence_tokens])
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        actions = [action.rule for action in instance.fields["actions"].field_list]
        agenda_actions = [actions[i] for i in agenda]
        world = instance.fields["world"].as_tensor({})
        expected_agenda_actions = world.get_agenda_for_sentence(sentence, add_paths_to_agenda=True)
        assert expected_agenda_actions == agenda_actions
