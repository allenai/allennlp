# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import RateCalculusDatasetReader
from allennlp.semparse.worlds import RateCalculusWorld

expected_sample_actions = {
        0: ['@start@ -> b',
            'b -> [<n,<n,b>>, n, n]',
            '<n,<n,b>> -> Equals',
            'n -> [<o,<d,n>>, o, d]',
            '<o,<d,n>> -> Value',
            'o -> o1',
            'd -> Dollar',
            'n -> 20'],
        1: ['@start@ -> b',
            'b -> [<b,<b,b>>, b, b]',
            '<b,<b,b>> -> And',
            'b -> [<n,<n,b>>, n, n]',
            '<n,<n,b>> -> Equals',
            'n -> [<o,<d,n>>, o, d]',
            '<o,<d,n>> -> Value',
            'o -> o1',
            'd -> Unit',
            'n -> 5',
            'b -> [<b,<b,b>>, b, b]',
            '<b,<b,b>> -> And',
            'b -> [<n,<n,b>>, n, n]',
            '<n,<n,b>> -> Equals',
            'n -> [<o,<d,<d,n>>>, o, d, d]',
            '<o,<d,<d,n>>> -> Rate',
            'o -> o2',
            'd -> Dollar',
            'd -> Unit',
            'n -> 10',
            'b -> [<b,<b,b>>, b, b]',
            '<b,<b,b>> -> And',
            'b -> [<n,<n,b>>, n, n]',
            '<n,<n,b>> -> Equals',
            'n -> [<o,<d,n>>, o, d]',
            '<o,<d,n>> -> Value',
            'o -> o2',
            'd -> Unit',
            'n -> 3',
            'b -> [<b,<b,b>>, b, b]',
            '<b,<b,b>> -> And',
            'b -> [<o,<o,b>>, o, o]',
            '<o,<o,b>> -> IsPart',
            'o -> o1',
            'o -> o0',
            'b -> [<b,<b,b>>, b, b]',
            '<b,<b,b>> -> And',
            'b -> [<o,<o,b>>, o, o]',
            '<o,<o,b>> -> IsPart',
            'o -> o2',
            'o -> o0',
            'b -> [<n,<n,b>>, n, n]',
            '<n,<n,b>> -> Equals',
            'n -> [<o,<d,n>>, o, d]',
            '<o,<d,n>> -> Value',
            'o -> o0',
            'd -> Dollar',
            'n -> q1']
}


def assert_sample_instances_correct(instance, correct_actions):
    # The content of this will be tested indirectly by checking the actions; we'll just make
    # sure we get a RateCalculusWorld object in here.
    assert isinstance(instance.fields['world'].as_tensor({}), RateCalculusWorld)

    action_fields = instance.fields['actions'].field_list
    actions = [action_field.rule for action_field in action_fields]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).  This is good, because
    # sometimes DPD does silly things that we don't want to reproduce.  But it also means if we
    # break something, we might not notice in the test unless we check this explicitly.
    #num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    # assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples, which is _not_ the first one
    # in the file.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]

    assert actions == correct_actions


def assert_alg514_instances_correct(instance, correct_actions):
    # The content of this will be tested indirectly by checking the actions; we'll just make
    # sure we get a RateCalculusWorld object in here.
    assert isinstance(instance.fields['world'].as_tensor({}), RateCalculusWorld)

    action_fields = instance.fields['actions'].field_list
    actions = [action_field.rule for action_field in action_fields]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).  This is good, because
    # sometimes DPD does silly things that we don't want to reproduce.  But it also means if we
    # break something, we might not notice in the test unless we check this explicitly.
    #num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    # assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples, which is _not_ the first one
    # in the file.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]

    assert actions == correct_actions

def assert_sample_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 2

    for i, instance in enumerate(instances):
        assert_sample_instances_correct(instance, expected_sample_actions[i])

def assert_sample_alg514_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 9

    #for i, instance in enumerate(instances):
    #    assert_alg514_instances_correct(instance, expected_alg514_actions[i])


class RateCalculusDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads_sample_data(self):
        reader = RateCalculusDatasetReader(lazy=False)
        dataset = reader.read(str(self.FIXTURES_ROOT / "data" / "ratecalculus" / "sample_data.json"))
        assert_sample_dataset_correct(dataset)

    def test_reader_reads_alg514(self):
        reader = RateCalculusDatasetReader(lazy=False)
        filename = "sample_alg514_binaryAnd_unitDims_simplifiedVars.json"
        filepath = str(self.FIXTURES_ROOT / "data" / "ratecalculus" / filename)
        dataset = reader.read(filepath)
        assert_sample_alg514_dataset_correct(dataset)

    def test_reader_reads_all_alg514(self):
        reader = RateCalculusDatasetReader(lazy=False)
        filename = "alg514_canonical.json"
        filepath = str(self.FIXTURES_ROOT / "data" / "ratecalculus" / filename)
        dataset = reader.read(filepath)
        assert(len(list(dataset)) >= 504)
