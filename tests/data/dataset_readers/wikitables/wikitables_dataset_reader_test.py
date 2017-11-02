# pylint: disable=no-self-use
from allennlp.data.dataset_readers import WikitablesDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestWikitablesDatasetReader(AllenNlpTestCase):
    def test_reading_works(self):
        tables_directory = "tests/fixtures/data/wikitables"
        dpd_output_directory = "tests/fixtures/data/wikitables/dpd_output"
        reader = WikitablesDatasetReader(tables_directory, dpd_output_directory)
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
        assert len(dataset.instances) == 2
        action_sequence = dataset.instances[0].fields["action_sequences"].field_list[0]
        actions = [l.label for l in action_sequence.field_list]
        assert actions == ['@@START@@', 'd', 'd -> [<d,d>, d]', '<d,d> -> M0', 'd -> [<e,d>, e]',
                           '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> R',
                           '<d,e> -> D1', 'e -> [<r,e>, r]', '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                           '<<#1,#2>,<#2,#1>> -> R', '<e,r> -> C0', 'r -> [<e,r>, e]', '<e,r> -> C1',
                           'e -> cell:usl_a_league', '@@END@@']
