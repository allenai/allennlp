# pylint: disable=no-self-use
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.data.semparse.worlds import WikiTablesWorld
from allennlp.common.testing import AllenNlpTestCase


class TestWikiTablesDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        tables_directory = "tests/fixtures/data/wikitables"
        dpd_output_directory = "tests/fixtures/data/wikitables/dpd_output"
        reader = WikiTablesDatasetReader(tables_directory, dpd_output_directory)
        dataset = reader.read("tests/fixtures/data/wikitables/sample_data.examples")
        assert len(dataset.instances) == 2
        instance = dataset.instances[0]
        question_tokens = ["what", "was", "the", "last", "year", "where", "this", "team", "was",
                           "a", "part", "of", "the", "usl", "a", "-", "league", "?"]
        assert [t.text for t in instance.fields["question"].tokens] == question_tokens
        entities = instance.fields['table'].knowledge_graph.get_all_entities()
        assert len(entities) == 47
        assert 'fb:row.row.year' in entities

        # The content of this will be tested indirectly by checking the actions; we'll just make
        # sure we get a WikiTablesWorld object in here.
        assert isinstance(instance.fields['world'].as_tensor({}), WikiTablesWorld)

        actions = [action_field.rule for action_field in instance.fields['actions'].field_list]
        assert len(actions) == 186

        # This is going to be long, but I think it's worth it, to be sure that all of the actions
        # we're expecting are present, and there are no extras.
        assert sorted(actions) == [
                # Placeholder types
                "<#1,#1> -> !=",
                "<#1,#1> -> fb:type.object.type",
                "<#1,#1> -> var",
                "<#1,<#1,#1>> -> and",
                "<#1,<#1,#1>> -> or",
                "<#1,d> -> count",
                "<<#1,#2>,<#2,#1>> -> reverse",

                # These complex types are largely just here to build up a few specific functions:
                # argmin and argmax.  The lambdas inside this are probably unnecessary, but are a
                # side-effect of how we add lambdas wherever we can.
                "<<d,d>,d> -> ['lambda x', d]",
                "<<d,d>,d> -> [<d,<<d,d>,d>>, d]",
                "<<d,e>,e> -> ['lambda x', e]",
                "<<d,e>,e> -> [<e,<<d,e>,e>>, e]",
                "<<d,p>,p> -> ['lambda x', p]",
                "<<d,p>,p> -> [<p,<<d,p>,p>>, p]",
                "<<d,r>,r> -> ['lambda x', r]",
                "<<d,r>,r> -> [<r,<<d,r>,r>>, r]",
                "<d,<<d,d>,d>> -> ['lambda x', <<d,d>,d>]",
                "<d,<<d,d>,d>> -> [<d,<d,<<d,d>,d>>>, d]",
                "<d,<d,<#1,<<d,#1>,#1>>>> -> argmax",
                "<d,<d,<#1,<<d,#1>,#1>>>> -> argmin",
                "<d,<d,<<d,d>,d>>> -> ['lambda x', <d,<<d,d>,d>>]",
                "<d,<d,<<d,d>,d>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
                "<d,<d,d>> -> -",
                "<d,<d,d>> -> ['lambda x', <d,d>]",
                "<d,<e,<<d,e>,e>>> -> ['lambda x', <e,<<d,e>,e>>]",
                "<d,<e,<<d,e>,e>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
                "<d,<p,<<d,p>,p>>> -> ['lambda x', <p,<<d,p>,p>>]",
                "<d,<p,<<d,p>,p>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",
                "<d,<r,<<d,r>,r>>> -> ['lambda x', <r,<<d,r>,r>>]",
                "<d,<r,<<d,r>,r>>> -> [<d,<d,<#1,<<d,#1>,#1>>>>, d]",

                # Operations that manipulate numbers.
                # TODO(mattg): question for Pradeep: why aren't these two-argument functions?
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
                # TODO(mattg): isn't this backwards?  Aren't these functions that take cells and
                # return numbers?
                "<d,e> -> fb:cell.cell.date",
                "<d,e> -> fb:cell.cell.num2",
                "<d,e> -> fb:cell.cell.number",
                "<d,p> -> ['lambda x', p]",
                "<d,p> -> [<<#1,#2>,<#2,#1>>, <p,d>]",
                "<d,r> -> ['lambda x', r]",
                "<d,r> -> [<<#1,#2>,<#2,#1>>, <r,d>]",
                "<d,r> -> fb:row.row.index",

                # Now we get to the CELL_TYPE, which is represented by "e".
                "<e,<<d,e>,e>> -> ['lambda x', <<d,e>,e>]",
                "<e,<<d,e>,e>> -> [<d,<e,<<d,e>,e>>>, d]",
                "<e,<e,<e,d>>> -> ['lambda x', <e,<e,d>>]",
                # TODO(mattg): why is this so complicated?
                "<e,<e,<e,d>>> -> date",
                "<e,<e,d>> -> ['lambda x', <e,d>]",
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
                # table.
                # TODO(mattg): this also seems backwards.
                "<e,r> -> fb:row.row.avg_attendance",
                "<e,r> -> fb:row.row.division",
                "<e,r> -> fb:row.row.league",
                "<e,r> -> fb:row.row.open_cup",
                "<e,r> -> fb:row.row.playoffs",
                "<e,r> -> fb:row.row.regular_season",
                "<e,r> -> fb:row.row.year",

                # PART_TYPE rules.
                # TODO(mattg): what is a cell part?
                "<p,<<d,p>,p>> -> ['lambda x', <<d,p>,p>]",
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
                "<r,<<d,r>,r>> -> ['lambda x', <<d,r>,r>]",
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

        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        actions = [actions[i] for i in action_indices]
        assert actions == ['@START@ -> d', 'd -> [<d,d>, d]', '<d,d> -> max', 'd -> [<e,d>, e]',
                           '<e,d> -> [<<#1,#2>,<#2,#1>>, <d,e>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<d,e> -> fb:cell.cell.date', 'e -> [<r,e>, r]',
                           '<r,e> -> [<<#1,#2>,<#2,#1>>, <e,r>]', '<<#1,#2>,<#2,#1>> -> reverse',
                           '<e,r> -> fb:row.row.year', 'r -> [<e,r>, e]', '<e,r> -> fb:row.row.league',
                           'e -> fb:cell.usl_a_league']
