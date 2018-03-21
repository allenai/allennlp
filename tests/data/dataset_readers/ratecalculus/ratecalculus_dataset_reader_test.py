# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import RateCalculusDatasetReader
from allennlp.data.semparse.worlds import RateCalculusWorld

def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 1
    instance = instances[0]

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
    num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    #assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples, which is _not_ the first one
    # in the file.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]
    assert actions == [
            '@START@ -> d',
            'd -> [<nd,nd>, d]',
            '<nd,nd> -> max',
            'd -> [<c,d>, c]',
            '<c,d> -> [<<#1,#2>,<#2,#1>>, <d,c>]',
            '<<#1,#2>,<#2,#1>> -> reverse',
            '<d,c> -> fb:cell.cell.date',
            'c -> [<r,c>, r]',
            '<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
            '<<#1,#2>,<#2,#1>> -> reverse',
            '<c,r> -> fb:row.row.year',
            'r -> [<c,r>, c]',
            '<c,r> -> fb:row.row.league',
            'c -> fb:cell.usl_a_league'
            ]


class RateCalculusDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        reader = RateCalculusDatasetReader(lazy=False)
        dataset = reader.read("tests/fixtures/data/ratecalculus/sample_data.json")
        assert_dataset_correct(dataset)

