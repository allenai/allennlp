"""
An ``AtisSqlTableContext`` represents the SQL context in which an utterance appears
for the Atis dataset, with the grammar and the valid actions.
"""
from typing import List, Dict, Tuple
import sqlite3
from copy import deepcopy

from parsimonious.grammar import Grammar

from allennlp.common.file_utils import cached_path
from allennlp.semparse.contexts.sql_context_utils import initialize_valid_actions, format_grammar_string, \
        format_action

# This is the base definition of the SQL grammar in a simplified sort of
# EBNF notation, and represented as a dictionary. The keys are the nonterminals and the values
# are the possible expansions of the nonterminal where each element in the list is one possible expansion.
# Rules that differ only in capitalization of keywords are mapped to the same action by
# the ``SqlVisitor``.  The nonterminal of the first rule is the starting symbol.
# In addition to the grammar here, we add ``col_ref``, ``table_name`` based on the tables
# that ``SqlTableContext`` is initialized with. ``number`` is initialized to
# be empty and later on updated based on the utterances. ``biexpr`` is altered based on the
# database to column references with strings that are allowed to appear in that column.
# We then create additional nonterminals for each column that may be used as a string constraint
# in the query.
# For example, to include city names as strings:
#
#       grammar_dictionary['biexpr'] = \
#               ['( "city" ws "." ws "city_name"  binop ws city_city_name_strings )',  ...
#       grammar_dictionary['city_city_name_strings'] = ['"NASHVILLE"', '"BOSTON"',  ...

GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY['statement'] = ['query ws ";" ws']
GRAMMAR_DICTIONARY['query'] = ['(ws "(" ws "SELECT" ws distinct ws select_results ws '
                               '"FROM" ws table_refs ws where_clause ws group_by_clause ws ")" ws)',
                               '(ws "(" ws "SELECT" ws distinct ws select_results ws '
                               '"FROM" ws table_refs ws where_clause ws ")" ws)',
                               '(ws "SELECT" ws distinct ws select_results ws '
                               '"FROM" ws table_refs ws where_clause ws)']
GRAMMAR_DICTIONARY['select_results'] = ['col_refs', 'agg']
GRAMMAR_DICTIONARY['agg'] = ['( agg_func ws "(" ws col_ref ws ")" )', '(agg_func ws "(" ws col ws ")" )']
GRAMMAR_DICTIONARY['agg_func'] = ['"MIN"', '"min"', '"MAX"', '"max"', '"COUNT"', '"count"']
GRAMMAR_DICTIONARY['col_refs'] = ['(col_ref ws "," ws col_refs)', '(col_ref)']
GRAMMAR_DICTIONARY['table_refs'] = ['(table_name ws "," ws table_refs)', '(table_name)']
GRAMMAR_DICTIONARY['where_clause'] = ['("WHERE" ws "(" ws conditions ws ")" ws)', '("WHERE" ws conditions ws)']
GRAMMAR_DICTIONARY['group_by_clause'] = ['("GROUP" ws "BY" ws col_ref)']
GRAMMAR_DICTIONARY['conditions'] = ['(condition ws conj ws conditions)',
                                    '(condition ws conj ws "(" ws conditions ws ")")',
                                    '("(" ws conditions ws ")" ws conj ws conditions)',
                                    '("(" ws conditions ws ")")',
                                    '("not" ws conditions ws )',
                                    '("NOT" ws conditions ws )',
                                    'condition']
GRAMMAR_DICTIONARY['condition'] = ['in_clause', 'ternaryexpr', 'biexpr']
GRAMMAR_DICTIONARY['in_clause'] = ['(ws col_ref ws "IN" ws query ws)']
GRAMMAR_DICTIONARY['biexpr'] = ['( col_ref ws binaryop ws value)', '(value ws binaryop ws value)']
GRAMMAR_DICTIONARY['binaryop'] = ['"+"', '"-"', '"*"', '"/"', '"="',
                                  '">="', '"<="', '">"', '"<"', '"is"', '"IS"']
GRAMMAR_DICTIONARY['ternaryexpr'] = ['(col_ref ws "not" ws "BETWEEN" ws value ws "AND" ws value ws)',
                                     '(col_ref ws "NOT" ws "BETWEEN" ws value ws "AND" ws value ws)',
                                     '(col_ref ws "BETWEEN" ws value ws "AND" ws value ws)']
GRAMMAR_DICTIONARY['value'] = ['("not" ws pos_value)', '("NOT" ws pos_value)', '(pos_value)']
GRAMMAR_DICTIONARY['pos_value'] = ['("ALL" ws query)', '("ANY" ws query)', 'number',
                                   'boolean', 'col_ref', 'agg_results', '"NULL"']
GRAMMAR_DICTIONARY['agg_results'] = ['(ws "("  ws "SELECT" ws distinct ws agg ws '
                                     '"FROM" ws table_name ws where_clause ws ")" ws)',
                                     '(ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws)']
GRAMMAR_DICTIONARY['boolean'] = ['"true"', '"false"']
GRAMMAR_DICTIONARY['ws'] = [r'~"\s*"i']
GRAMMAR_DICTIONARY['conj'] = ['"AND"', '"OR"']
GRAMMAR_DICTIONARY['distinct'] = ['("DISTINCT")', '("")']
GRAMMAR_DICTIONARY['number'] = ['""']

KEYWORDS = ['"SELECT"', '"FROM"', '"MIN"', '"MAX"', '"COUNT"', '"WHERE"', '"NOT"', '"IN"', '"LIKE"',
            '"IS"', '"BETWEEN"', '"AND"', '"ALL"', '"ANY"', '"NULL"', '"OR"', '"DISTINCT"']

NUMERIC_NONTERMINALS = ['number', 'time_range_start', 'time_range_end',
                        'fare_round_trip_cost', 'fare_one_direction_cost',
                        'flight_number', 'day_number', 'month_number', 'year_number']

class AtisSqlTableContext:
    """
    An ``AtisSqlTableContext`` represents the SQL context with a grammar of SQL and the valid actions
    based on the schema of the tables that it represents.

    Parameters
    ----------
    all_tables: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables in the dataset, the keys are the names of the tables
        that map to lists of the table's column names.
    tables_with_strings: ``Dict[str, List[str]]``
        A dictionary representing the SQL tables that we want to generate strings for. The keys are the
        names of the tables that map to lists of the table's column names.
    database_file : ``str``, optional
        The directory to find the sqlite database file. We query the sqlite database to find the strings
        that are allowed.
    """
    def __init__(self,
                 all_tables: Dict[str, List[str]] = None,
                 tables_with_strings: Dict[str, List[str]] = None,
                 database_file: str = None) -> None:
        self.all_tables = all_tables
        self.tables_with_strings = tables_with_strings
        if database_file:
            self.database_file = cached_path(database_file)
            self.connection = sqlite3.connect(self.database_file)
            self.cursor = self.connection.cursor()

        grammar_dictionary, strings_list = self.create_grammar_dict_and_strings()
        self.grammar_dictionary: Dict[str, List[str]] = grammar_dictionary
        self.strings_list: List[Tuple[str, str]] = strings_list

        self.grammar_string: str = self.get_grammar_string()
        self.grammar: Grammar = Grammar(self.grammar_string)
        self.valid_actions: Dict[str, List[str]] = initialize_valid_actions(self.grammar, KEYWORDS)
        if database_file:
            self.connection.close()

    def get_grammar_dictionary(self) -> Dict[str, List[str]]:
        return self.grammar_dictionary

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def create_grammar_dict_and_strings(self) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
        grammar_dictionary = deepcopy(GRAMMAR_DICTIONARY)
        strings_list = []

        if self.all_tables:
            grammar_dictionary['table_name'] = \
                    sorted([f'"{table}"'
                            for table in list(self.all_tables.keys())], reverse=True)
            grammar_dictionary['col_ref'] = ['"*"', 'agg']
            all_columns = []
            for table, columns in self.all_tables.items():
                grammar_dictionary['col_ref'].extend([f'("{table}" ws "." ws "{column}")'
                                                      for column in columns])
                all_columns.extend(columns)
            grammar_dictionary['col_ref'] = sorted(grammar_dictionary['col_ref'], reverse=True)
            grammar_dictionary['col'] = sorted([f'"{column}"' for column in all_columns], reverse=True)

        biexprs = []
        if self.tables_with_strings:
            for table, columns in self.tables_with_strings.items():
                biexprs.extend([f'("{table}" ws "." ws "{column}" ws binaryop ws {table}_{column}_string)'
                                for column in columns])
                for column in columns:
                    self.cursor.execute(f'SELECT DISTINCT {table} . {column} FROM {table}')
                    results = self.cursor.fetchall()

                    # Almost all the query values are in the database, we hardcode the rare case here.
                    if table == 'flight' and column == 'airline_code':
                        results.append(('EA',))
                    strings_list.extend([(format_action(f"{table}_{column}_string",
                                                        str(row[0]),
                                                        is_string=not 'number' in column,
                                                        is_number='number' in column),
                                          str(row[0]))
                                         for row in results])

                    if column.endswith('number'):
                        grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"{str(row[0])}"' for row in results], reverse=True)
                    else:
                        grammar_dictionary[f'{table}_{column}_string'] = \
                                sorted([f'"\'{str(row[0])}\'"' for row in results], reverse=True)

        grammar_dictionary['biexpr'] = sorted(biexprs, reverse=True) + \
                ['( col_ref ws binaryop ws value)', '(value ws binaryop ws value)']
        return grammar_dictionary, strings_list

    def get_grammar_string(self):
        return format_grammar_string(self.grammar_dictionary)
