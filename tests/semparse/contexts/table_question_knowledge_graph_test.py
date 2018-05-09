# pylint: disable=no-self-use,invalid-name,protected-access,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.semparse.contexts import TableQuestionKnowledgeGraph


class TestTableQuestionKnowledgeGraph(AllenNlpTestCase):
    def test_read_from_json_handles_simple_cases(self):
        json = {
                'question': [Token(x) for x in ['where', 'is', 'mersin', '?']],
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:cell.mersin'])
        assert graph.entities == ['-1', '0', '1', 'fb:cell.edirne', 'fb:cell.lake_gala',
                                  'fb:cell.mersin', 'fb:cell.paradeniz', 'fb:row.row.location',
                                  'fb:row.row.name_in_english']
        assert neighbors == {'fb:row.row.location'}
        neighbors = set(graph.neighbors['fb:row.row.name_in_english'])
        assert neighbors == {'fb:cell.paradeniz', 'fb:cell.lake_gala'}
        assert graph.entity_text['fb:cell.edirne'] == 'Edirne'
        assert graph.entity_text['fb:cell.lake_gala'] == 'Lake Gala'
        assert graph.entity_text['fb:cell.mersin'] == 'Mersin'
        assert graph.entity_text['fb:cell.paradeniz'] == 'Paradeniz'
        assert graph.entity_text['fb:row.row.location'] == 'Location'
        assert graph.entity_text['fb:row.row.name_in_english'] == 'Name in English'

        # These are default numbers that should always be in the graph.
        assert graph.neighbors['-1'] == []
        assert graph.neighbors['0'] == []
        assert graph.neighbors['1'] == []
        assert graph.entity_text['-1'] == '-1'
        assert graph.entity_text['0'] == '0'
        assert graph.entity_text['1'] == '1'

    def test_read_from_json_replaces_newlines(self):
        # The csv -> tsv conversion renders '\n' as r'\n' (with a literal slash character), that
        # gets read in a two characters instead of one.  We need to make sure we convert it back to
        # one newline character, so our splitting and other processing works correctly.
        json = {
                'question': [Token(x) for x in ['where', 'is', 'mersin', '?']],
                'columns': ['Name\\nin English', 'Location'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake\\nGala', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.entities == ['-1', '0', '1', 'fb:cell.edirne', 'fb:cell.lake_gala',
                                  'fb:cell.mersin', 'fb:cell.paradeniz', 'fb:part.gala',
                                  'fb:part.lake', 'fb:part.paradeniz', 'fb:row.row.location',
                                  'fb:row.row.name_in_english']
        assert graph.entity_text['fb:row.row.name_in_english'] == 'Name\nin English'

    def test_read_from_json_splits_columns_when_necessary(self):
        json = {
                'question': [Token(x) for x in ['where', 'is', 'mersin', '?']],
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Mersin with spaces'],
                          ['Lake, Gala', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.entities == ['-1', '0', '1', 'fb:cell.edirne', 'fb:cell.lake_gala',
                                  'fb:cell.mersin_with_spaces', 'fb:cell.paradeniz', 'fb:part.gala',
                                  'fb:part.lake', 'fb:part.paradeniz', 'fb:row.row.location',
                                  'fb:row.row.name_in_english']
        assert graph.neighbors['fb:part.lake'] == []
        assert graph.neighbors['fb:part.gala'] == []
        assert graph.neighbors['fb:part.paradeniz'] == []

    def test_read_from_json_handles_numbers_in_question(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                'question': [Token(x) for x in ['one', '4']],
                'columns': [],
                'cells': []
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.neighbors['1'] == []
        assert graph.neighbors['4'] == []
        assert graph.entity_text['1'] == 'one'
        assert graph.entity_text['4'] == '4'

    def test_get_cell_parts_returns_cell_text_on_simple_cells(self):
        assert TableQuestionKnowledgeGraph._get_cell_parts('Team') == [('fb:part.team', 'Team')]
        assert TableQuestionKnowledgeGraph._get_cell_parts('2006') == [('fb:part.2006', '2006')]
        assert TableQuestionKnowledgeGraph._get_cell_parts('Wolfe Tones') == [('fb:part.wolfe_tones',
                                                                               'Wolfe Tones')]

    def test_get_cell_parts_splits_on_commas(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts('United States, Los Angeles')
        assert set(parts) == {('fb:part.united_states', 'United States'),
                              ('fb:part.los_angeles', 'Los Angeles')}

    def test_get_cell_parts_on_past_failure_cases(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts('Checco D\'Angelo\n "Jimmy"')
        assert set(parts) == {('fb:part.checco_d_angelo', "Checco D\'Angelo"),
                              ('fb:part._jimmy', '"Jimmy"')}

    def test_get_cell_parts_handles_multiple_splits(self):
        parts = TableQuestionKnowledgeGraph._get_cell_parts('this, has / lots\n of , commas')
        assert set(parts) == {('fb:part.this', 'this'),
                              ('fb:part.has', 'has'),
                              ('fb:part.lots', 'lots'),
                              ('fb:part.of', 'of'),
                              ('fb:part.commas', 'commas')}

    def test_should_split_column_returns_false_when_all_text_is_simple(self):
        assert TableQuestionKnowledgeGraph._should_split_column_cells(['Team', '2006', 'Wolfe Tones']) is False

    def test_should_split_column_returns_true_when_one_input_is_splitable(self):
        assert TableQuestionKnowledgeGraph._should_split_column_cells(['Team, 2006', 'Wolfe Tones']) is True

    def test_read_from_json_handles_diacritics(self):
        json = {
                'question': [],
                'columns': ['Name in English', 'Name in Turkish', 'Location'],
                'cells': [['Lake Van', 'Van Gölü', 'Mersin'],
                          ['Lake Gala', 'Gala Gölü', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.name_in_turkish'])
        assert neighbors == {'fb:cell.van_golu', 'fb:cell.gala_golu'}

        json = {
                'question': [],
                'columns': ['Notes'],
                'cells': [['Ordained as a priest at\nReșița on March, 29th 1936']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.notes'])
        assert neighbors == {'fb:cell.ordained_as_a_priest_at_resita_on_march_29th_1936'}

        json = {
                'question': [],
                'columns': ['Player'],
                'cells': [['Mateja Kežman']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.player'])
        assert neighbors == {'fb:cell.mateja_kezman'}

        json = {
                'question': [],
                'columns': ['Venue'],
                'cells': [['Arena Națională, Bucharest, Romania']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.venue'])
        assert neighbors == {'fb:cell.arena_nationala_bucharest_romania'}

    def test_read_from_json_handles_newlines_in_columns(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                'question': [],
                'columns': ['Peak\\nAUS', 'Peak\\nNZ'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.peak_aus'])
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(graph.neighbors['fb:row.row.peak_nz'])
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(graph.neighbors['fb:cell.1'])
        assert neighbors == {'fb:row.row.peak_aus'}

        json = {
                'question': [],
                'columns': ['Title'],
                'cells': [['Dance of the\\nSeven Veils']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.title'])
        assert neighbors == {'fb:cell.dance_of_the_seven_veils'}

    def test_read_from_json_handles_diacritics_and_newlines(self):
        json = {
                'question': [],
                'columns': ['Notes'],
                'cells': [['8 districts\nFormed from Orūzgān Province in 2004']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.notes'])
        assert neighbors == {'fb:cell.8_districts_formed_from_oruzgan_province_in_2004'}

    def test_read_from_json_handles_crazy_unicode(self):
        json = {
                'question': [],
                'columns': ['Town'],
                'cells': [['Viðareiði'],
                          ['Funningsfjørður'],
                          ['Froðba']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.town'])
        assert neighbors == {
                'fb:cell.funningsfj_r_ur',
                'fb:cell.vi_arei_i',
                'fb:cell.fro_ba',
                }

        json = {
                'question': [],
                'columns': ['Fate'],
                'cells': [['Sunk at 45°00′N 11°21′W﻿ / ﻿45.000°N 11.350°W'],
                          ['66°22′32″N 29°20′19″E﻿ / ﻿66.37556°N 29.33861°E']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.fate'])
        assert neighbors == {'fb:cell.sunk_at_45_00_n_11_21_w_45_000_n_11_350_w',
                             'fb:cell.66_22_32_n_29_20_19_e_66_37556_n_29_33861_e'}

        json = {
                'question': [],
                'columns': ['€0.01', 'Σ Points'],
                'cells': [['6,000', '9.5']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row._0_01'])
        assert neighbors == {'fb:cell.6_000'}
        neighbors = set(graph.neighbors['fb:row.row._points'])
        assert neighbors == {'fb:cell.9_5'}

        json = {
                'question': [],
                'columns': ['Division'],
                'cells': [['1ª Aut. Pref.']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.division'])
        assert neighbors == {'fb:cell.1_aut_pref'}

    def test_read_from_json_handles_parentheses_correctly(self):
        json = {
                'question': [],
                'columns': ['Urban settlements'],
                'cells': [['Dzhebariki-Khaya\\n(Джебарики-Хая)'],
                          ['South Korea (KOR)'],
                          ['Area (km²)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.urban_settlements'])
        assert neighbors == {'fb:cell.dzhebariki_khaya',
                             'fb:cell.south_korea_kor',
                             'fb:cell.area_km'}

        json = {
                'question': [],
                'columns': ['Margin\\nof victory'],
                'cells': [['−9 (67-67-68-69=271)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.margin_of_victory'])
        assert neighbors == {'fb:cell._9_67_67_68_69_271'}

        json = {
                'question': [],
                'columns': ['Record'],
                'cells': [['4.08 m (13 ft 41⁄2 in)']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.record'])
        assert neighbors == {'fb:cell.4_08_m_13_ft_41_2_in'}

    def test_read_from_json_handles_columns_with_duplicate_normalizations(self):
        json = {
                'question': [],
                'columns': ['# of votes', '% of votes'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row._of_votes'])
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(graph.neighbors['fb:row.row._of_votes_2'])
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(graph.neighbors['fb:cell.1'])
        assert neighbors == {'fb:row.row._of_votes'}

    def test_read_from_json_handles_cells_with_duplicate_normalizations(self):
        json = {
                'question': [],
                'columns': ['answer'],
                'cells': [['yes'], ['yes*'], ['yes'], ['yes '], ['yes*']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)

        # There are three unique text strings that all normalize to "yes", so there are three
        # fb:cell.yes entities.  Hopefully we produce them in the same order as SEMPRE does...
        assert graph.entities == ['-1', '0', '1', 'fb:cell.yes', 'fb:cell.yes_2', 'fb:cell.yes_3',
                                  'fb:row.row.answer']

    def test_get_numbers_from_tokens_works_for_arabic_numerals(self):
        tokens = [Token(x) for x in ['7', '1.0', '-20']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [('7', '7'), ('1.000', '1.0'), ('-20', '-20')]

    def test_get_numbers_from_tokens_works_for_ordinal_and_cardinal_numbers(self):
        tokens = [Token(x) for x in ['one', 'five', 'Seventh']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [('1', 'one'), ('5', 'five'), ('7', 'Seventh')]

    def test_get_numbers_from_tokens_works_for_months(self):
        tokens = [Token(x) for x in ['January', 'March', 'october']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [('1', 'January'), ('3', 'March'), ('10', 'october')]

    def test_get_numbers_from_tokens_works_for_units(self):
        tokens = [Token(x) for x in ['1ghz', '3.5mm', '-2m/s']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [('1', '1ghz'), ('3.500', '3.5mm'), ('-2', '-2m/s')]

    def test_get_numbers_from_tokens_works_with_magnitude_words(self):
        tokens = [Token(x) for x in ['one', 'million', '7', 'thousand']]
        numbers = TableQuestionKnowledgeGraph._get_numbers_from_tokens(tokens)
        assert numbers == [('1000000', 'one million'), ('7000', '7 thousand')]

    def test_get_linked_agenda_items(self):
        json = {
                'question': [Token(x) for x in ['where', 'is', 'mersin', '?']],
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph.get_linked_agenda_items() == ['fb:cell.mersin', 'fb:row.row.location']

    def test_get_longest_span_matching_entities(self):
        json = {
                'question': [Token(x) for x in ['where', 'is', 'lake', 'big', 'gala', '?']],
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Lake Big'],
                          ['Lake Big Gala', 'Edirne']]
                }
        graph = TableQuestionKnowledgeGraph.read_from_json(json)
        assert graph._get_longest_span_matching_entities() == ['fb:cell.lake_big_gala']
