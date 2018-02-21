# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.data.semparse.worlds import WikiTablesWorld

def assert_dataset_correct(dataset):
    # pylint is having trouble applying the disable command above to this method; maybe because
    # it is so long?
    # pylint: disable=no-self-use,protected-access
    instances = list(dataset)
    assert len(instances) == 2
    instance = instances[0]

    assert instance.fields.keys() == {'question', 'table', 'world', 'actions', 'target_action_sequences'}

    question_tokens = ["what", "was", "the", "last", "year", "where", "this", "team", "was", "a",
                       "part", "of", "the", "usl", "a", "-", "league", "?"]
    assert [t.text for t in instance.fields["question"].tokens] == question_tokens

    entities = instance.fields['table'].knowledge_graph.entities
    assert len(entities) == 47
    assert sorted(entities) == [
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
    action_fields.sort(key=lambda x: x.rule)
    assert len(action_fields) == 173

    # Here we're making sure that we're deciding which things are "nonterminals" correctly
    # (where "nonterminal" in this setting means "part of the global grammar", including things
    # that are actually terminal productions...).  So there are two blocks that we're looking
    # for: the "<e,r> -> fb:row.row.[column]" block, and the "e -> fb:cell.[cell]" block.  Each
    # of these has a null entity thrown in that we treat as a "nonterminal", because it's part
    # of the global grammar.

    # Before the "<e,r> -> fb:row.row.[column]" block.
    for i in range(54):
        assert action_fields[i]._right_is_nonterminal is True, f"{i}, {action_fields[i].rule}"
    # Start of the "<e,r> -> fb:row.row.[column]" block.
    for i in range(54, 57):
        assert action_fields[i]._right_is_nonterminal is False, f"{i}, {action_fields[i].rule}"
    # This is the null column, right in the middle of the other columns.
    assert action_fields[57]._right_is_nonterminal is True, f"{i}, {action_fields[i].rule}"
    # End of the "<e,r> -> fb:row.row.[column]" block.
    for i in range(58, 62):
        assert action_fields[i]._right_is_nonterminal is False, f"{i}, {action_fields[i].rule}"
    # In between the column and the cell blocks.
    for i in range(62, 116):
        assert action_fields[i]._right_is_nonterminal is True, f"{i}, {action_fields[i].rule}"
    # Start of the "e -> fb:cell.[column]" block.
    for i in range(116, 151):
        assert action_fields[i]._right_is_nonterminal is False, f"{i}, {action_fields[i].rule}"
    # This is the null cell, right in the middle of the other cells.
    assert action_fields[151]._right_is_nonterminal is True, f"{i}, {action_fields[i].rule}"
    # End of the "e -> fb:cell.[column]" block.
    for i in range(152, 157):
        assert action_fields[i]._right_is_nonterminal is False, f"{i}, {action_fields[i].rule}"
    # After the "e -> fb:cell.[column]" block.
    for i in range(157, 173):
        assert action_fields[i]._right_is_nonterminal is True, f"{i}, {action_fields[i].rule}"

    # This is going to be long, but I think it's worth it, to be sure that all of the actions
    # we're expecting are present, and there are no extras.
    actions = [action_field.rule for action_field in action_fields]
    assert actions == [
            # Placeholder types
            "<#1,#1> -> !=",
            "<#1,#1> -> fb:type.object.type",
            "<#1,<#1,#1>> -> and",
            "<#1,<#1,#1>> -> or",
            "<#1,d> -> count",
            "<<#1,#2>,<#2,#1>> -> reverse",

            # These complex types are largely just here to build up a few specific functions:
            # argmin and argmax.
            "<<d,d>,d> -> [<d,<<d,d>,d>>, d]",
            "<<d,e>,e> -> [<e,<<d,e>,e>>, e]",
            "<<d,p>,p> -> [<p,<<d,p>,p>>, p]",
            "<<d,r>,r> -> [<r,<<d,r>,r>>, r]",
            "<d,<<d,d>,d>> -> [<d,<d,<<d,d>,d>>>, d]",
            "<d,<d,<#1,<<d,#1>,#1>>>> -> argmax",
            "<d,<d,<#1,<<d,#1>,#1>>>> -> argmin",
            "<d,<d,<<d,d>,d>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
            "<d,<d,d>> -> -",
            "<d,<e,<<d,e>,e>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
            "<d,<p,<<d,p>,p>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
            "<d,<r,<<d,r>,r>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",

            # Operations that manipulate numbers.
            # Note that the comparison operators here are single-argument functions, taking a
            # date or number and returning the set of all dates or numbers in the context that
            # are [comparator] what was given.
            "<d,d> -> <",
            "<d,d> -> <=",
            "<d,d> -> >",
            "<d,d> -> >=",
            "<d,d> -> ['lambda x', d]",
            "<d,d> -> [<#1,<#1,#1>>, d]",
            "<d,d> -> [<<#1,#2>,<#2,#1>>, <d,d>]",
            "<d,d> -> [<d,<d,d>>, d]",
            # These are single-argument functions because the "d" type here actually represents
            # a set, and we're performing some aggregation over the set and returning a single
            # number (actually a set with a single number in it).
            "<d,d> -> avg",
            "<d,d> -> max",
            "<d,d> -> min",
            "<d,d> -> sum",
            "<d,e> -> ['lambda x', e]",
            "<d,e> -> [<<#1,#2>,<#2,#1>>, <e,d>]",
            # These might look backwards, but that's because SEMPRE chose to make them
            # backwards.  fb:a.b is a function that takes b and returns a.  So
            # fb:cell.cell.date takes cell.date and returns cell and fb:row.row.index takes
            # row.index and returns row.
            "<d,e> -> fb:cell.cell.date",
            "<d,e> -> fb:cell.cell.num2",
            "<d,e> -> fb:cell.cell.number",
            "<d,p> -> ['lambda x', p]",
            "<d,p> -> [<<#1,#2>,<#2,#1>>, <p,d>]",
            "<d,r> -> ['lambda x', r]",
            "<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]",
            "<d,r> -> fb:row.row.index",

            # Now we get to the CELL_TYPE, which is represented by "e".
            "<e,<<d,e>,e>> -> [<d,<e,<<d,e>,e>>>, d]",
            # "date" is a function that takes three numbers: (date 2018 01 06).  And these
            # numbers have type "e", not type "d", for some reason.
            "<e,<e,<e,d>>> -> date",
            "<e,<e,d>> -> [<e,<e,<e,d>>>, e]",
            "<e,d> -> ['lambda x', d]",
            "<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]",
            "<e,d> -> [<e,<e,d>>, e]",
            "<e,d> -> number",
            "<e,e> -> ['lambda x', e]",
            "<e,e> -> [<#1,<#1,#1>>, e]",
            "<e,e> -> [<<#1,#2>,<#2,#1>>, <e,e>]",
            "<e,p> -> ['lambda x', p]",
            "<e,p> -> [<<#1,#2>,<#2,#1>>, <p,e>]",
            "<e,r> -> ['lambda x', r]",
            "<e,r> -> [<<#1,#2>,<#2,#1>>, <r,e>]",

            # These are instance-specific production rules.  These are the columns in the
            # table.  Remember that SEMPRE did things backwards: fb:row.row.division takes a
            # cell ID and returns the row that has that cell in its row.division column.  This
            # is why we have to reverse all of these functions to go from a row to the cell in
            # a particular column.
            "<e,r> -> fb:row.row.avg_attendance",
            "<e,r> -> fb:row.row.division",
            "<e,r> -> fb:row.row.league",
            "<e,r> -> fb:row.row.null",  # null column, representing an empty set
            "<e,r> -> fb:row.row.open_cup",
            "<e,r> -> fb:row.row.playoffs",
            "<e,r> -> fb:row.row.regular_season",
            "<e,r> -> fb:row.row.year",

            # PART_TYPE rules.  A cell part is for when a cell has text that can be split into
            # multiple parts.  We don't really handle this, so we don't have any terminal
            # productions here.  We actually skip all logical forms that have "fb:part"
            # productions, and we'll never actually push one of these non-terminals onto our
            # stack.  But they're in the grammar, so we they are in our list of valid actions.
            "<p,<<d,p>,p>> -> [<d,<p,<<d,p>,p>>>, d]",
            "<p,d> -> ['lambda x', d]",
            "<p,d> -> [<<#1,#2>,<#2,#1>>, <d,p>]",
            "<p,e> -> ['lambda x', e]",
            "<p,e> -> [<<#1,#2>,<#2,#1>>, <e,p>]",
            "<p,e> -> fb:cell.cell.part",
            "<p,p> -> ['lambda x', p]",
            "<p,p> -> [<#1,<#1,#1>>, p]",
            "<p,p> -> [<<#1,#2>,<#2,#1>>, <p,p>]",
            "<p,r> -> ['lambda x', r]",
            "<p,r> -> [<<#1,#2>,<#2,#1>>, <r,p>]",

            # Functions that operate on rows.
            "<r,<<d,r>,r>> -> [<d,<r,<<d,r>,r>>>, d]",
            "<r,d> -> ['lambda x', d]",
            "<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]",
            "<r,e> -> ['lambda x', e]",
            "<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]",
            "<r,p> -> ['lambda x', p]",
            "<r,p> -> [<<#1,#2>,<#2,#1>>, <p,r>]",
            "<r,r> -> ['lambda x', r]",
            "<r,r> -> [<#1,<#1,#1>>, r]",
            "<r,r> -> [<<#1,#2>,<#2,#1>>, <r,r>]",
            "<r,r> -> fb:row.row.next",

            # All valid start types - we just say that any BASIC_TYPE is a valid start type.
            "@START@ -> d",
            "@START@ -> e",
            "@START@ -> p",
            "@START@ -> r",

            # Now we get to productions for the basic types.  The first part of each section
            # here has rules for expanding the basic type with function applications; after
            # that we get to terminal productions (and a "BASIC_TYPE -> x" rule for lambda
            # variable productions).

            # Date and numbers.  We don't have any terminal productions for these, as for some
            # reason terminal numbers are actually represented as CELL_TYPE.
            # TODO(mattg,pradeep): we should probably change the number type to not be the same
            # as CELL_TYPE.
            "d -> [<#1,#1>, d]",
            "d -> [<#1,d>, d]",
            "d -> [<#1,d>, e]",
            "d -> [<#1,d>, p]",
            "d -> [<#1,d>, r]",
            "d -> [<<d,d>,d>, <d,d>]",
            "d -> [<d,d>, d]",
            "d -> [<e,d>, e]",
            "d -> [<p,d>, p]",
            "d -> [<r,d>, r]",
            "d -> x",

            # CELL_TYPE productions.  We have some numbers here, whatever numbers showed up in
            # the question (as digits), as well as some which are hard-coded to cover
            # ordinals and cardinals that are written out as text.
            "e -> -1",
            "e -> 0",
            "e -> 1",
            "e -> 2",
            "e -> 3",
            "e -> 4",
            "e -> 5",
            "e -> 6",
            "e -> 7",
            "e -> 8",
            "e -> 9",
            "e -> [<#1,#1>, e]",
            "e -> [<<d,e>,e>, <d,e>]",
            "e -> [<d,e>, d]",
            "e -> [<e,e>, e]",
            "e -> [<p,e>, p]",
            "e -> [<r,e>, r]",
            # And these are the cells that we saw in the table.
            "e -> fb:cell.10_727",
            "e -> fb:cell.11th",
            "e -> fb:cell.1st",
            "e -> fb:cell.1st_round",
            "e -> fb:cell.1st_western",
            "e -> fb:cell.2",
            "e -> fb:cell.2001",
            "e -> fb:cell.2002",
            "e -> fb:cell.2003",
            "e -> fb:cell.2004",
            "e -> fb:cell.2005",
            "e -> fb:cell.2006",
            "e -> fb:cell.2007",
            "e -> fb:cell.2008",
            "e -> fb:cell.2009",
            "e -> fb:cell.2010",
            "e -> fb:cell.2nd",
            "e -> fb:cell.2nd_pacific",
            "e -> fb:cell.2nd_round",
            "e -> fb:cell.3rd_pacific",
            "e -> fb:cell.3rd_round",
            "e -> fb:cell.3rd_usl_3rd",
            "e -> fb:cell.4th_round",
            "e -> fb:cell.4th_western",
            "e -> fb:cell.5_575",
            "e -> fb:cell.5_628",
            "e -> fb:cell.5_871",
            "e -> fb:cell.5th",
            "e -> fb:cell.6_028",
            "e -> fb:cell.6_260",
            "e -> fb:cell.6_851",
            "e -> fb:cell.7_169",
            "e -> fb:cell.8_567",
            "e -> fb:cell.9_734",
            "e -> fb:cell.did_not_qualify",
            "e -> fb:cell.null",  # null cell, representing an empty set
            "e -> fb:cell.quarterfinals",
            "e -> fb:cell.semifinals",
            "e -> fb:cell.usl_a_league",
            "e -> fb:cell.usl_first_division",
            "e -> fb:cell.ussf_d_2_pro_league",
            "e -> x",

            # We don't have any table-specific productions for PART_TYPE or ROW_TYPE, so these
            # are just function applications, and one terminal production for rows.
            "p -> [<#1,#1>, p]",
            "p -> [<<d,p>,p>, <d,p>]",
            "p -> [<d,p>, d]",
            "p -> [<e,p>, e]",
            "p -> [<p,p>, p]",
            "p -> [<r,p>, r]",
            "p -> x",
            "r -> [<#1,#1>, r]",
            "r -> [<<d,r>,r>, <d,r>]",
            "r -> [<d,r>, d]",
            "r -> [<e,r>, e]",
            "r -> [<p,r>, p]",
            "r -> [<r,r>, r]",
            "r -> fb:type.row",
            "r -> x",
            ]

    # We should have been able to read all of the logical forms in the file.  If one of them can't
    # be parsed, or the action sequences can't be mapped correctly, the DatasetReader will skip the
    # logical form, log an error, and keep going (i.e., it won't crash).  This is good, because
    # sometimes DPD does silly things that we don't want to reproduce.  But it also means if we
    # break something, we might not notice in the test unless we check this explicitly.
    num_action_sequences = len(instance.fields["target_action_sequences"].field_list)
    assert num_action_sequences == 10

    # We should have sorted the logical forms by length.  This is the action sequence
    # corresponding to the shortest logical form in the examples, which is _not_ the first one
    # in the file.
    action_sequence = instance.fields["target_action_sequences"].field_list[0]
    action_indices = [l.sequence_index for l in action_sequence.field_list]
    actions = [actions[i] for i in action_indices]
    assert actions == [
            '@START@ -> d',
            'd -> [<d,d>, d]',
            '<d,d> -> max',
            'd -> [<e,d>, e]',
            '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]',
            '<<#1,#2>,<#2,#1>> -> reverse',
            '<d,e> -> fb:cell.cell.date',
            'e -> [<r,e>, r]',
            '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
            '<<#1,#2>,<#2,#1>> -> reverse',
            '<e,r> -> fb:row.row.year',
            'r -> [<e,r>, e]',
            '<e,r> -> fb:row.row.league',
            'e -> fb:cell.usl_a_league'
            ]


class WikiTablesDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        tables_directory = "tests/fixtures/data/wikitables"
        dpd_output_directory = "tests/fixtures/data/wikitables/dpd_output"
        reader = WikiTablesDatasetReader(lazy=False,
                                         tables_directory=tables_directory,
                                         dpd_output_directory=dpd_output_directory)
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
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
                '@START@ -> d',
                'd -> [<e,d>, e]',
                '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<d,e> -> fb:cell.cell.date',
                'e -> [<r,e>, r]',
                '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<e,r> -> fb:row.row.year',
                'r -> [<d,r>, d]',
                '<d,r> -> fb:row.row.index',
                'd -> [<d,d>, d]',
                '<d,d> -> max',
                'd -> [<r,d>, r]',
                '<r,d> -> [<<#1,#2>,<#2,#1>>, <d,r>]',
                '<<#1,#2>,<#2,#1>> -> reverse',
                '<d,r> -> fb:row.row.index',
                'r -> [<e,r>, e]',
                '<e,r> -> fb:row.row.league',
                'e -> fb:cell.usl_a_league'
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
