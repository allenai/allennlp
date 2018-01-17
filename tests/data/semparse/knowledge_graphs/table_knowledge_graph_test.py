# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph


class TestTableKnowledgeGraph(AllenNlpTestCase):
    def test_read_from_json_handles_simple_cases(self):
        json = {
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:cell.mersin'])
        assert graph.entities == ['fb:cell.edirne', 'fb:cell.lake_gala', 'fb:cell.mersin',
                                  'fb:cell.paradeniz', 'fb:row.row.location',
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

    def test_read_from_json_handles_diacritics(self):
        json = {
                'columns': ['Name in English', 'Name in Turkish', 'Location'],
                'cells': [['Lake Van', 'Van Gölü', 'Mersin'],
                          ['Lake Gala', 'Gala Gölü', 'Edirne']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.name_in_turkish'])
        assert neighbors == {'fb:cell.van_golu', 'fb:cell.gala_golu'}

        json = {
                'columns': ['Notes'],
                'cells': [['Ordained as a priest at\nReșița on March, 29th 1936']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.notes'])
        assert neighbors == {'fb:cell.ordained_as_a_priest_at_resita_on_march_29th_1936'}

        json = {
                'columns': ['Player'],
                'cells': [['Mateja Kežman']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.player'])
        assert neighbors == {'fb:cell.mateja_kezman'}

        json = {
                'columns': ['Venue'],
                'cells': [['Arena Națională, Bucharest, Romania']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.venue'])
        assert neighbors == {'fb:cell.arena_nationala_bucharest_romania'}

    def test_read_from_json_handles_newlines_in_columns(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                'columns': ['Peak\\nAUS', 'Peak\\nNZ'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.peak_aus'])
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(graph.neighbors['fb:row.row.peak_nz'])
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(graph.neighbors['fb:cell.1'])
        assert neighbors == {'fb:row.row.peak_aus'}

        json = {
                'columns': ['Title'],
                'cells': [['Dance of the\\nSeven Veils']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.title'])
        assert neighbors == {'fb:cell.dance_of_the_seven_veils'}

    def test_read_from_json_handles_diacritics_and_newlines(self):
        json = {
                'columns': ['Notes'],
                'cells': [['8 districts\nFormed from Orūzgān Province in 2004']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.notes'])
        assert neighbors == {'fb:cell.8_districts_formed_from_oruzgan_province_in_2004'}

    def test_read_from_json_handles_crazy_unicode(self):
        json = {
                'columns': ['Town'],
                'cells': [['Viðareiði'],
                          ['Funningsfjørður'],
                          ['Froðba']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.town'])
        assert neighbors == {'fb:cell.funningsfj_r_ur',
                             'fb:cell.vi_arei_i',
                             'fb:cell.fro_ba',
                             }

        json = {
                'columns': ['Fate'],
                'cells': [['Sunk at 45°00′N 11°21′W﻿ / ﻿45.000°N 11.350°W'],
                          ['66°22′32″N 29°20′19″E﻿ / ﻿66.37556°N 29.33861°E']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.fate'])
        assert neighbors == {'fb:cell.sunk_at_45_00_n_11_21_w_45_000_n_11_350_w',
                             'fb:cell.66_22_32_n_29_20_19_e_66_37556_n_29_33861_e'}

        json = {
                'columns': ['€0.01'],
                'cells': [['6,000']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row._0_01'])
        assert neighbors == {'fb:cell.6_000'}

        json = {
                'columns': ['Division'],
                'cells': [['1ª Aut. Pref.']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.division'])
        assert neighbors == {'fb:cell.1_aut_pref'}

    def test_read_from_json_handles_parentheses_correctly(self):
        json = {
                'columns': ['Urban settlements'],
                'cells': [['Dzhebariki-Khaya\\n(Джебарики-Хая)'],
                          ['South Korea (KOR)'],
                          ['Area (km²)']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.urban_settlements'])
        assert neighbors == {'fb:cell.dzhebariki_khaya',
                             'fb:cell.south_korea_kor',
                             'fb:cell.area_km'}

        json = {
                'columns': ['Margin\\nof victory'],
                'cells': [['−9 (67-67-68-69=271)']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.margin_of_victory'])
        assert neighbors == {'fb:cell._9_67_67_68_69_271'}

        json = {
                'columns': ['Record'],
                'cells': [['4.08 m (13 ft 41⁄2 in)']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row.record'])
        assert neighbors == {'fb:cell.4_08_m_13_ft_41_2_in'}

    def test_read_from_json_handles_columns_with_duplicate_normalizations(self):
        json = {
                'columns': ['# of votes', '% of votes'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(graph.neighbors['fb:row.row._of_votes'])
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(graph.neighbors['fb:row.row._of_votes_2'])
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(graph.neighbors['fb:cell.1'])
        assert neighbors == {'fb:row.row._of_votes'}
