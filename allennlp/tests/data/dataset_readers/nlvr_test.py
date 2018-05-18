# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import NlvrDatasetReader
from allennlp.semparse.worlds import NlvrWorld


class TestNlvrDatasetReader(AllenNlpTestCase):
    def test_reader_reads_ungrouped_data(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "nlvr" /
                        "sample_ungrouped_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        assert instance.fields.keys() == {'sentence', 'agenda', 'worlds', 'actions', 'labels',
                                          'identifier'}
        sentence_tokens = instance.fields["sentence"].tokens
        expected_tokens = ['There', 'is', 'a', 'circle', 'closely', 'touching', 'a', 'corner', 'of',
                           'a', 'box', '.']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields["actions"].field_list]
        assert len(actions) == 115
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        agenda_strings = [actions[rule_id] for rule_id in agenda]
        assert set(agenda_strings) == set(['<o,o> -> circle',
                                           '<o,t> -> object_exists',
                                           '<o,o> -> touch_corner'])
        worlds = [world_field.as_tensor({}) for world_field in instance.fields["worlds"].field_list]
        assert isinstance(worlds[0], NlvrWorld)
        label = instance.fields["labels"].field_list[0].label
        assert label == "true"

    def test_agenda_indices_are_correct(self):
        reader = NlvrDatasetReader()
        test_file = str(self.FIXTURES_ROOT / "data" / "nlvr" /
                        "sample_ungrouped_data.jsonl")
        dataset = reader.read(test_file)
        instances = list(dataset)
        instance = instances[0]
        sentence_tokens = instance.fields["sentence"].tokens
        sentence = " ".join([t.text for t in sentence_tokens])
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        actions = [action.rule for action in instance.fields["actions"].field_list]
        agenda_actions = [actions[i] for i in agenda]
        world = instance.fields["worlds"].field_list[0].as_tensor({})
        expected_agenda_actions = world.get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
        assert expected_agenda_actions == agenda_actions

    def test_reader_reads_grouped_data(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "nlvr" /
                        "sample_grouped_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]
        assert instance.fields.keys() == {'sentence', 'agenda', 'worlds', 'actions', 'labels',
                                          'identifier'}
        sentence_tokens = instance.fields["sentence"].tokens
        expected_tokens = ['There', 'is', 'a', 'circle', 'closely', 'touching', 'a', 'corner', 'of',
                           'a', 'box', '.']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields["actions"].field_list]
        assert len(actions) == 115
        agenda = [item.sequence_index for item in instance.fields["agenda"].field_list]
        agenda_strings = [actions[rule_id] for rule_id in agenda]
        assert set(agenda_strings) == set(['<o,o> -> circle',
                                           '<o,o> -> touch_corner',
                                           '<o,t> -> object_exists'
                                          ])
        worlds = [world_field.as_tensor({}) for world_field in instance.fields["worlds"].field_list]
        assert all([isinstance(world, NlvrWorld) for world in worlds])
        labels = [label.label for label in instance.fields["labels"].field_list]
        assert labels == ["true", "false", "true", "false"]

    def test_reader_reads_processed_data(self):
        # Processed data contains action sequences that yield the correct denotations, obtained from
        # an offline search.
        test_file = str(self.FIXTURES_ROOT / "data" / "nlvr" /
                        "sample_processed_data.jsonl")
        dataset = NlvrDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]
        assert instance.fields.keys() == {"sentence", "target_action_sequences",
                                          "worlds", "actions", "labels", "identifier"}
        all_action_sequence_indices = instance.fields["target_action_sequences"].field_list
        assert len(all_action_sequence_indices) == 20
        action_sequence_indices = [item.sequence_index for item in
                                   all_action_sequence_indices[0].field_list]
        actions = [action.rule for action in instance.fields["actions"].field_list]
        action_sequence = [actions[rule_id] for rule_id in action_sequence_indices]
        assert action_sequence == ['@start@ -> t',
                                   't -> [<o,t>, o]',
                                   '<o,t> -> object_exists',
                                   'o -> [<o,o>, o]',
                                   '<o,o> -> touch_corner',
                                   'o -> [<o,o>, o]',
                                   '<o,o> -> circle',
                                   'o -> all_objects']
