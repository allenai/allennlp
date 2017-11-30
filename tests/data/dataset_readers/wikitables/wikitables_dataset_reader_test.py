# pylint: disable=no-self-use
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.common.testing import AllenNlpTestCase


class TestWikiTablesDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        tables_directory = "tests/fixtures/data/wikitables"
        dpd_output_directory = "tests/fixtures/data/wikitables/dpd_output"
        reader = WikiTablesDatasetReader(tables_directory, dpd_output_directory)
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
        assert len(dataset.instances) == 2
        instance = dataset.instances[0]
        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        actions = [l.label for l in action_sequence.field_list]
        question_tokens = ["what", "was", "the", "last", "year", "where", "this", "team", "was",
                           "a", "part", "of", "the", "usl", "a", "-", "league", "?"]
        assert [t.text for t in instance.fields["question"].tokens] == question_tokens
        assert actions == ['@@START@@', 'd', 'd -> [<d,d>, d]', '<d,d> -> max', 'd -> [<e,d>, e]',
                           '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<d,e> -> fb:cell.cell.date', 'e -> [<r,e>, r]',
                           '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<e,r> -> fb:row.row.year', 'r -> [<e,r>, e]', '<e,r> -> fb:row.row.league',
                           'e -> fb:cell.usl_a_league', '@@END@@']
        entities = instance.fields['table'].knowledge_graph.get_all_entities()
        assert len(entities) == 47
        assert 'fb:row.row.year' in entities
