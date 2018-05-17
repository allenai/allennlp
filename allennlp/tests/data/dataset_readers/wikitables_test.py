# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.semparse.worlds import WikiTablesWorld


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 2
    instance = instances[0]

    assert instance.fields.keys() == {
            'question',
            'table',
            'world',
            'actions',
            'target_action_sequences',
            'example_lisp_string',
            }

    question_tokens = ["what", "was", "the", "last", "year", "where", "this", "team", "was", "a",
                       "part", "of", "the", "usl", "a", "-", "league", "?"]
    assert [t.text for t in instance.fields["question"].tokens] == question_tokens

    entities = instance.fields['table'].knowledge_graph.entities
    assert len(entities) == 59
    assert sorted(entities) == [
            # Numbers, which are represented as graph entities, as we link them to the question.
            '-1',
            '0',
            '1',

            # The table cell entity names.
            'fb:cell.10_727',
            'fb:cell.11th',
            'fb:cell.1st',
            'fb:cell.1st_round',
            'fb:cell.1st_western',
            'fb:cell.2',
            'fb:cell.2001',
            'fb:cell.2002',
            'fb:cell.2003',
            'fb:cell.2004',
            'fb:cell.2005',
            'fb:cell.2006',
            'fb:cell.2007',
            'fb:cell.2008',
            'fb:cell.2009',
            'fb:cell.2010',
            'fb:cell.2nd',
            'fb:cell.2nd_pacific',
            'fb:cell.2nd_round',
            'fb:cell.3rd_pacific',
            'fb:cell.3rd_round',
            'fb:cell.3rd_usl_3rd',
            'fb:cell.4th_round',
            'fb:cell.4th_western',
            'fb:cell.5_575',
            'fb:cell.5_628',
            'fb:cell.5_871',
            'fb:cell.5th',
            'fb:cell.6_028',
            'fb:cell.6_260',
            'fb:cell.6_851',
            'fb:cell.7_169',
            'fb:cell.8_567',
            'fb:cell.9_734',
            'fb:cell.did_not_qualify',
            'fb:cell.quarterfinals',
            'fb:cell.semifinals',
            'fb:cell.usl_a_league',
            'fb:cell.usl_first_division',
            'fb:cell.ussf_d_2_pro_league',

            # Cell parts
            'fb:part.11th',
            'fb:part.1st',
            'fb:part.2nd',
            'fb:part.3rd',
            'fb:part.4th',
            'fb:part.5th',
            'fb:part.pacific',
            'fb:part.usl_3rd',
            'fb:part.western',

            # Column headers
            'fb:row.row.avg_attendance',
            'fb:row.row.division',
            'fb:row.row.league',
            'fb:row.row.open_cup',
            'fb:row.row.playoffs',
            'fb:row.row.regular_season',
            'fb:row.row.year',
            ]

    # The content of this will be tested indirectly by checking the actions; we'll just make
    # sure we get a WikiTablesWorld object in here.
    assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesWorld)

    action_fields = instance.fields['actions'].field_list
    actions = [action_field.rule for action_field in action_fields]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).  This is good, because
    # sometimes DPD does silly things that we don't want to reproduce.  But it also means if we
    # break something, we might not notice in the test unless we check this explicitly.
    num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples _by tree size_, which is _not_ the
    # first one in the file, or the shortest logical form by _string length_.  It's also a totally
    # made up logical form, just to demonstrate that we're sorting things correctly.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]
    assert actions == [
            '@start@ -> r',
            'r -> [<c,r>, c]',
            '<c,r> -> fb:row.row.league',
            'c -> fb:cell.3rd_usl_3rd'
            ]


class WikiTablesDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        params = {
                'lazy': False,
                'tables_directory': "tests/fixtures/data/wikitables",
                'dpd_output_directory': "tests/fixtures/data/wikitables/dpd_output",
                }
        reader = WikiTablesDatasetReader.from_params(Params(params))
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
        assert_dataset_correct(dataset)

    def test_reader_reads_preprocessed_file(self):
        # We're should get the exact same results when reading a pre-processed file as we get when
        # we read the original data.
        reader = WikiTablesDatasetReader()
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data_preprocessed.jsonl")
        assert_dataset_correct(dataset)

    def test_read_respects_max_dpd_tries_when_not_sorting(self):
        tables_directory = "tests/fixtures/data/wikitables"
        dpd_output_directory = "tests/fixtures/data/wikitables/dpd_output"
        reader = WikiTablesDatasetReader(lazy=False,
                                         sort_dpd_logical_forms=False,
                                         max_dpd_logical_forms=1,
                                         max_dpd_tries=1,
                                         tables_directory=tables_directory,
                                         dpd_output_directory=dpd_output_directory)
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
        instances = list(dataset)
        instance = instances[0]
        actions = [action_field.rule for action_field in instance.fields['actions'].field_list]

        # We should have just taken the first logical form from the file, which has the following
        # action sequence.
        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        action_strings = [actions[i] for i in action_indices]
        assert action_strings == [
                '@start@ -> d',
                'd -> [<c,d>, c]',
                '<c,d> -> [<<#1,#2>,<#2,#1>>, <d,c>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<d,c> -> fb:cell.cell.date',
                'c -> [<r,c>, r]',
                '<r,c> -> [<<#1,#2>,<#2,#1>>, <c,r>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<c,r> -> fb:row.row.year',
                'r -> [<n,r>, n]',
                '<n,r> -> fb:row.row.index',
                'n -> [<nd,nd>, n]',
                '<nd,nd> -> max',
                'n -> [<r,n>, r]',
                '<r,n> -> [<<#1,#2>,<#2,#1>>, <n,r>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<n,r> -> fb:row.row.index',
                'r -> [<c,r>, c]',
                '<c,r> -> fb:row.row.league',
                'c -> fb:cell.usl_a_league'
                ]

    def test_parse_example_line(self):
        # pylint: disable=no-self-use,protected-access
        with open("tests/fixtures/data/wikitables/sample_data.examples") as filename:
            lines = filename.readlines()
        example_info = WikiTablesDatasetReader._parse_example_line(lines[0])
        question = 'what was the last year where this team was a part of the usl a-league?'
        assert example_info == {'id': 'nt-0',
                                'question': question,
                                'table_filename': 'tables/590.csv'}
