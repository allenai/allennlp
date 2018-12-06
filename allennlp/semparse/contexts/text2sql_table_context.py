# pylint: disable=anomalous-backslash-in-string
"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
"""
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import TableColumn

GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY["statement"] = ['(query ws ";")', '(query ws)']
GRAMMAR_DICTIONARY["query"] = ['(ws select_core ws groupby_clause ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause ws orderby_clause)',
                               '(ws select_core ws groupby_clause ws limit)',
                               '(ws select_core ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause)',
                               '(ws select_core ws orderby_clause)',
                               '(ws select_core)']

GRAMMAR_DICTIONARY["select_core"] = ['(select_with_distinct ws select_results ws from_clause ws where_clause)',
                                     '(select_with_distinct ws select_results ws from_clause)',
                                     '(select_with_distinct ws select_results ws where_clause)',
                                     '(select_with_distinct ws select_results)']
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(ws "SELECT" ws "DISTINCT")', '(ws "SELECT")']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['"*"', '(table_name ws ".*")',
                                       '(expr ws "AS" wsp name)', 'expr', '(col_ref ws "AS" wsp name)']

GRAMMAR_DICTIONARY["from_clause"] = ['ws "FROM" ws source']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['source_table', 'source_subq']
GRAMMAR_DICTIONARY["source_table"] = ['(table_name ws "AS" wsp name)', 'table_name']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" ws "AS" ws name)', '("(" ws query ws ")")']
GRAMMAR_DICTIONARY["limit"] = ['("LIMIT" ws "1")', '("LIMIT" ws number)']

GRAMMAR_DICTIONARY["where_clause"] = ['(ws "WHERE" wsp "(" ws expr ws where_conj ws ")" ws where_conj ws )', '(ws "WHERE" wsp "(" ws expr ws where_conj ws ")")', '(ws "WHERE" wsp "(" ws expr ws  ")" ws where_conj)', '(ws "WHERE" wsp "(" ws expr ws ")")', '(ws "WHERE" wsp expr ws where_conj)', '(ws "WHERE" wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(ws "AND" wsp expr ws where_conj)', '(ws "AND" wsp expr)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(ws "GROUP" ws "BY" ws group_clause ws "HAVING" ws expr)',
                                        '(ws "GROUP" ws "BY" ws group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," ws group_clause)', '(ws expr)']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ws "ORDER" ws "BY" ws order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," ws order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ws ordering)', '(ws expr)']
GRAMMAR_DICTIONARY["ordering"] = ['(ws "ASC")', '(ws "DESC")']

GRAMMAR_DICTIONARY["col_ref"] = ['(table_name ws "." ws column_name)', 'table_name']
GRAMMAR_DICTIONARY["table_name"] = ['name']
GRAMMAR_DICTIONARY["column_name"] = ['name']
GRAMMAR_DICTIONARY["ws"] = ['~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = ['~"\s+"i']
GRAMMAR_DICTIONARY['name'] = ['~"[a-zA-Z]\w*"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr',
                              # Like expressions.
                              '(value wsp "LIKE" wsp string)',
                              # Between expressions.
                              '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                              # Binary expressions.
                              '(value ws binaryop wsp expr)',
                              # Unary expressions.
                              '(unaryop ws expr)',
                              # Two types of null check expressions.
                              '(col_ref ws "IS" ws "NOT" ws "NULL")',
                              '(col_ref ws "IS" ws "NULL")',
                              'source_subq',
                              'value']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp "NOT" wsp "IN" wsp string_set)',
                                 '(value wsp "IN" wsp string_set)',
                                 '(value wsp "NOT" wsp "IN" wsp expr)',
                                 '(value wsp "IN" wsp expr)']

GRAMMAR_DICTIONARY["value"] = ['parenval', '"YEAR(CURDATE())"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['(fname ws "(" ws "DISTINCT" ws arg_list_or_star ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")']

GRAMMAR_DICTIONARY["arg_list_or_star"] = ['arg_list', '"*"']
GRAMMAR_DICTIONARY["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
 # TODO(MARK): Massive hack, remove and modify the grammar accordingly
GRAMMAR_DICTIONARY["number"] = ['~"\d*\.?\d+"i', "'3'", "'4'"]
GRAMMAR_DICTIONARY["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
GRAMMAR_DICTIONARY["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY["fname"] = ['"COUNT"', '"SUM"', '"MAX"', '"MIN"', '"AVG"', '"ALL"']
GRAMMAR_DICTIONARY["boolean"] = ['"true"', '"false"']

# TODO(MARK): This is not tight enough. AND/OR are strictly boolean value operators.
GRAMMAR_DICTIONARY["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', '"AND"', '"OR"', '"LIKE"']
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"NOT"']



GLOBAL_DATASET_VALUES: Dict[str, List[str]] = {
        # These are used to check values are present, or numbers of authors.
        "scholar": ["0", "1", "2"],
        # 0 is used for "sea level", 750 is a "major" lake, and 150000 is a "major" city.
        "geography": ["0", "750", "150000"],
        # This defines what an "above average" restaurant is.
        "restaurants": ["2.5"]
}

GLOBAL_DATASET_VARIABLE_TYPES: Dict[str, Dict[str, Set[Tuple[str, str, str]]]] = {
        "geography": {'var0': {('RIVER', '.', 'RIVER_NAME'),
                               ('CITY', '.', 'CITY_NAME'),
                               ('MOUNTAIN', '.', 'MOUNTAIN_NAME'),
                               ('HIGHLOW', '.', 'HIGHEST_POINT'),
                               ('CITY', '.', 'STATE_NAME'),
                               ('HIGHLOW', '.', 'STATE_NAME'),
                               ('HIGHLOW', '.', 'LOWEST_POINT'),
                               ('STATE', '.', 'CAPITAL'),
                               ('BORDER_INFO', '.', 'STATE_NAME'),
                               ('RIVER', '.', 'COUNTRY_NAME'),
                               ('LAKE', '.', 'STATE_NAME'),
                               ('STATE', '.', 'STATE_NAME'),
                               ('RIVER', '.', 'TRAVERSE'),
                               ('BORDER_INFO', '.', 'BORDER'),
                               ('MOUNTAIN', '.', 'STATE_NAME')},
                      'var1': {('CITY', '.', 'STATE_NAME'),
                               ('STATE', '.', 'STATE_NAME'),
                               ('RIVER', '.', 'TRAVERSE'),
                               ('BORDER_INFO', '.', 'STATE_NAME')},
                      '750': {('LAKE', '.', 'AREA'),
                              ('RIVER', '.', 'LENGTH')},
                      '150000':{('CITY', '.', 'POPULATION')}
                     },
        "restaurants": {'city_name0': {('LOCATION', '.', 'CITY_NAME')},
                        'name0': {('RESTAURANT', '.', 'NAME')},
                        'region0': {('GEOGRAPHIC', '.', 'REGION')},
                        'food_type0': {('RESTAURANT', '.', 'FOOD_TYPE')},
                        'street_name0': {('LOCATION', '.', 'STREET_NAME')},
                        '2.5': {('RESTAURANT', '.', 'RATING')},
                        'county0': {('GEOGRAPHIC', '.', 'COUNTY')}
                       },

        "imdb": {'actor_birth_city0': {('ACTOR', '.', 'BIRTH_CITY')},
                 'writer_name0': {('WRITER', '.', 'NAME')},
                 'cast_role0': {('CAST', '.', 'ROLE')},
                 'movie_title0': {('MOVIE', '.', 'TITLE')},
                 'actor_birth_year0': {('ACTOR', '.', 'BIRTH_YEAR')},
                 'director_birth_city0': {('DIRECTOR', '.', 'BIRTH_CITY')},
                 'director_name0': {('DIRECTOR', '.', 'NAME')},
                 'actor_name0': {('ACTOR', '.', 'NAME')},
                 'movie_release_year0': {('MOVIE', '.', 'RELEASE_YEAR')},
                 'movie_release_year1': {('MOVIE', '.', 'RELEASE_YEAR')},
                 'actor_name1': {('ACTOR', '.', 'NAME')},
                 'genre_genre0': {('GENRE', '.', 'GENRE')},
                 'tv_series_title0': {('TV_SERIES', '.', 'TITLE')},
                 'actor_gender0': {('ACTOR', '.', 'GENDER')},
                 'company_name0': {('COMPANY', '.', 'NAME')},
                 'actor_nationality0': {('ACTOR', '.', 'NATIONALITY')},
                 'keyword_keyword0': {('KEYWORD', '.', 'KEYWORD')},
                 'producer_name0': {('PRODUCER', '.', 'NAME'), ('WRITER', '.', 'NAME')},
                 'director_nationality0': {('DIRECTOR', '.', 'NATIONALITY')},
                 'tv_series_release_year0': {('TV_SERIES', '.', 'RELEASE_YEAR')},
                 'director_gender0': {('DIRECTOR', '.', 'GENDER')}
                },
        "academic": {'author_name0': {('AUTHOR', '.', 'NAME')},
                     'publication_title0': {('PUBLICATION', '.', 'TITLE')},
                     'conference_name0': {('CONFERENCE', '.', 'NAME')},
                     'publication_year0': {('PUBLICATION', '.', 'YEAR')},
                     'publication_year1': {('PUBLICATION', '.', 'YEAR')},
                     'domain_name0': {('DOMAIN', '.', 'NAME')},
                     'organization_name0': {('ORGANIZATION', '.', 'NAME')},
                     'keyword_keyword0': {('KEYWORD', '.', 'KEYWORD')},
                     'journal_name0': {('JOURNAL', '.', 'NAME')},
                     'author_name1': {('AUTHOR', '.', 'NAME')},
                     'publication_citation_num0': {('PUBLICATION', '.', 'CITATION_NUM')},
                     'organization_continent0': {('ORGANIZATION', '.', 'CONTINENT')},
                     'conference_name1': {('CONFERENCE', '.', 'NAME')},
                     'author_name2': {('AUTHOR', '.', 'NAME')}
                    },

        "yelp": {'business_name0': {('BUSINESS', '.', 'NAME')},
                 'business_city0': {('BUSINESS', '.', 'CITY')},
                 'business_review_count0': {('BUSINESS', '.', 'REVIEW_COUNT')},
                 'category_name0': {('CATEGORY', '.', 'CATEGORY_NAME')},
                 'user_name0': {('USER', '.', 'NAME')},
                 'business_rating0': {('BUSINESS', '.', 'RATING')},
                 'business_state0': {('BUSINESS', '.', 'STATE')},
                 'tip_year0': {('TIP', '.', 'YEAR')},
                 'checkin_day0': {('CHECKIN', '.', 'DAY')},
                 'category_name1': {('CATEGORY', '.', 'CATEGORY_NAME')},
                 'review_year0': {('REVIEW', '.', 'YEAR')},
                 'review_rating0': {('REVIEW', '.', 'RATING')},
                 'tip_likes0': {('TIP', '.', 'LIKES')},
                 'neighbourhood_name0': {('NEIGHBOURHOOD', '.', 'NEIGHBOURHOOD_NAME')},
                 'review_month0': {('REVIEW', '.', 'MONTH')},
                 'tip_month0': {('TIP', '.', 'MONTH')}
                },
        "scholar": {'authorname0': {('AUTHOR', '.', 'AUTHORNAME')},
                    'authorname1': {('AUTHOR', '.', 'AUTHORNAME')},
                    'keyphrasename0': {('KEYPHRASE', '.', 'KEYPHRASENAME')},
                    'keyphrasename1': {('KEYPHRASE', '.', 'KEYPHRASENAME')},
                    'venuename0': {('VENUE', '.', 'VENUENAME')},
                    'venuename1': {('VENUE', '.', 'VENUENAME')},
                    'datasetname0': {('DATASET', '.', 'DATASETNAME')},
                    'datasetname1': {('DATASET', '.', 'DATASETNAME')},
                    'title0': {('PAPER', '.', 'TITLE')},
                    'journalname0': {('JOURNAL', '.', 'JOURNALNAME')},
                    '0': {('PAPER', '.', 'JOURNALID')},
                    'year0': {('PAPER', '.', 'YEAR')},
                    'venueid0': {('PAPER', '.', 'VENUEID')},
                   },

        "atis": {'airport_code0': {('AIRPORT', '.', 'AIRPORT_CODE'), ('AIRPORT_SERVICE', '.', 'AIRPORT_CODE')},
                 'city_name0': {('CITY', '.', 'CITY_NAME')},
                 'day_number0': {('DATE_DAY', '.', 'DAY_NUMBER')},
                 'month_number0': {('DATE_DAY', '.', 'MONTH_NUMBER')},
                 'year0': {('DATE_DAY', '.', 'YEAR')},
                 'city_name1': {('CITY', '.', 'CITY_NAME')},
                 'day_name0': {('DAYS', '.', 'DAY_NAME')},
                 'city_name2': {('CITY', '.', 'CITY_NAME')},
                 'airline_code0': {('AIRLINE', '.', 'AIRLINE_CODE'), ('FLIGHT', '.', 'AIRLINE_CODE')},
                 'class_type0': {('FARE_BASIS', '.', 'CLASS_TYPE')},
                 'city_name3': {('CITY', '.', 'CITY_NAME')},
                 'transport_type0': {('GROUND_SERVICE', '.', 'TRANSPORT_TYPE')},
                 'departure_time1': {('FLIGHT', '.', 'DEPARTURE_TIME')},
                 'flight_number0': {('FLIGHT', '.', 'FLIGHT_NUMBER')},
                 'round_trip_required0': {('FARE', '.', 'ROUND_TRIP_REQUIRED')},
                 'round_trip_cost0': {('FARE', '.', 'ROUND_TRIP_COST')},
                 'departure_time0': {('FLIGHT', '.', 'DEPARTURE_TIME')},
                 'arrival_time0': {('FLIGHT', '.', 'ARRIVAL_TIME'), ('FLIGHT_STOP', '.', 'ARRIVAL_TIME')},
                 'stops0': {('FLIGHT', '.', 'STOPS')},
                 'days_code0': {('DAYS', '.', 'DAYS_CODE')},
                 'arrival_time1': {('FLIGHT', '.', 'ARRIVAL_TIME')},
                 'restriction_code0': {('RESTRICTION', '.', 'RESTRICTION_CODE')},
                 'flight_days0': {('FLIGHT', '.', 'FLIGHT_DAYS')},
                 'airline_name0': {('AIRLINE', '.', 'AIRLINE_NAME')},
                 'fare_basis_code0': {('FARE', '.', 'FARE_BASIS_CODE'), ('FARE_BASIS', '.', 'FARE_BASIS_CODE')},
                 'meal_description0': {('FOOD_SERVICE', '.', 'MEAL_DESCRIPTION')},
                 'economy0': {('FARE_BASIS', '.', 'ECONOMY')},
                 'state_code0': {('CITY', '.', 'STATE_CODE'), ('STATE', '.', 'STATE_CODE')},
                 'airport_code1': {('AIRPORT', '.', 'AIRPORT_CODE')},
                 'aircraft_code0': {('AIRCRAFT', '.', 'AIRCRAFT_CODE')},
                 'basic_type0': {('AIRCRAFT', '.', 'BASIC_TYPE')},
                 'manufacturer0': {('AIRCRAFT', '.', 'MANUFACTURER')},
                 'state_name0': {('STATE', '.', 'STATE_NAME')},
                 'state_name1': {('STATE', '.', 'STATE_NAME')},
                 'day_name1': {('DAYS', '.', 'DAY_NAME')},
                 'flight_number1': {('FLIGHT', '.', 'FLIGHT_NUMBER')},
                 'time_elapsed0': {('FLIGHT', '.', 'TIME_ELAPSED')},
                 'airline_code1': {('AIRLINE', '.', 'AIRLINE_CODE'), ('FLIGHT', '.', 'AIRLINE_CODE')},
                 'connections0': {('FLIGHT', '.', 'CONNECTIONS')},
                 'booking_class0': {('CLASS_OF_SERVICE', '.', 'BOOKING_CLASS'),
                                    ('FARE_BASIS', '.', 'BOOKING_CLASS')},
                 'arrival_time2': {('FLIGHT', '.', 'ARRIVAL_TIME')},
                 'departure_time2': {('FLIGHT', '.', 'DEPARTURE_TIME')},
                 'departure_time3': {('FLIGHT', '.', 'DEPARTURE_TIME')},
                 'state_name2': {('STATE', '.', 'STATE_NAME')},
                 'day_name2': {('DAYS', '.', 'DAY_NAME')},
                 'day_name3': {('DAYS', '.', 'DAY_NAME')},
                 'day_name4': {('DAYS', '.', 'DAY_NAME')},
                 'one_direction_cost0': {('FARE', '.', 'ONE_DIRECTION_COST')},
                 'class_type1': {('FARE_BASIS', '.', 'CLASS_TYPE')},
                 'year1': {('DATE_DAY', '.', 'YEAR')},
                 'day_number1': {('DATE_DAY', '.', 'DAY_NUMBER')},
                 'meal_code0': {('FOOD_SERVICE', '.', 'MEAL_CODE'), ('FLIGHT', '.', 'MEAL_CODE')},
                 'airline_code2': {('FLIGHT', '.', 'AIRLINE_CODE')},
                 'country_name0': {('CITY', '.', 'COUNTRY_NAME')},
                 'fare_basis_code1': {('FARE', '.', 'FARE_BASIS_CODE')},
                 'propulsion0': {('AIRCRAFT', '.', 'PROPULSION')},
                 'airport_name0': {('AIRPORT', '.', 'AIRPORT_NAME')},
                 'discounted0': {('FARE_BASIS', '.', 'DISCOUNTED')},
                 'meal_code1': {('FLIGHT', '.', 'MEAL_CODE')},
                 'state_code1': {('CITY', '.', 'STATE_CODE')},
                 'booking_class1': {('CLASS_OF_SERVICE', '.', 'BOOKING_CLASS')},
                 'transport_type1': {('GROUND_SERVICE', '.', 'TRANSPORT_TYPE')}}
}


def update_grammar_with_tables(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, List[TableColumn]],
                               constrained: bool = False) -> None:
    table_names = sorted([f'"{table}"' for table in
                          list(schema.keys())], reverse=True)
    grammar_dictionary['table_name'] = table_names

    all_columns = set()
    for table_name, table in schema.items():
        if constrained:
            all_columns.update([f'("{table_name}" ws "." ws "{column.name}")' for column in table])
        else:
            all_columns.update([f'"{column.name}"' for column in table])

    sorted_columns = sorted(all_columns, reverse=True)
    if constrained:
        del grammar_dictionary['column_name']
        grammar_dictionary['col_ref'] = sorted_columns + ['table_name']
    else:
        grammar_dictionary['column_name'] = sorted_columns

def update_grammar_with_global_values(grammar_dictionary: Dict[str, List[str]], dataset_name: str):

    values = GLOBAL_DATASET_VALUES.get(dataset_name, [])
    values_for_grammar = [f'"{str(value)}"' for value in values]
    grammar_dictionary["value"] = values_for_grammar + grammar_dictionary["value"]


def update_grammar_to_be_variable_free(grammar_dictionary: Dict[str, List[str]]):
    """
    SQL is a predominately variable free language in terms of simple usage, in the
    sense that most queries do not create references to variables which are not
    already static tables in a dataset. However, it is possible to do this via
    derived tables. If we don't require this functionality, we can tighten the
    grammar, because we don't need to support aliased tables.
    """

    # Tables in variable free grammars cannot be aliased, so we
    # remove this functionality from the grammar.
    grammar_dictionary["select_result"] = ['"*"', '(table_name ws ".*")', 'expr']

    # Similarly, collapse the definition of a source table
    # to not contain aliases and modify references to subqueries.
    grammar_dictionary["single_source"] = ['table_name', '("(" ws query ws ")")']
    del grammar_dictionary["source_subq"]
    del grammar_dictionary["source_table"]

    grammar_dictionary["expr"] = ['in_expr',
                                  '(value wsp "LIKE" wsp string)',
                                  '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                                  '(value ws binaryop wsp expr)',
                                  '(unaryop ws expr)',
                                  '(col_ref ws "IS" ws "NOT" ws "NULL")',
                                  '(col_ref ws "IS" ws "NULL")',
                                  # This used to be source_subq - now
                                  # we don't need aliases, we can colapse it to queries.
                                  '("(" ws query ws ")")',
                                  'value']

    # Finally, remove the ability to reference an arbitrary name,
    # because now we don't have aliased tables, we don't need
    # to recognise new variables.
    del grammar_dictionary["name"]

def remove_number_and_string_types(grammar_dictionary: Dict[str, List[str]]) -> None:
    """
    Variables can be treated as numbers or strings if their type can be inferred -
    however, that can be difficult, so instead, we can just treat them all as values
    and be a bit looser on the typing we allow in our grammar. Here we just remove
    all references to number and string from the grammar, replacing them with value.
    """
    grammar_dictionary["string_set_vals"] = ['(value ws "," ws string_set_vals)', 'value']
    grammar_dictionary["value"].remove('string')
    grammar_dictionary["value"].remove('number')
    grammar_dictionary["limit"] = ['("LIMIT" ws "1")', '("LIMIT" ws value)']
    grammar_dictionary["expr"][1] = '(value wsp "LIKE" wsp value)'
    del grammar_dictionary["string"]
    del grammar_dictionary["number"]

# TODO(Mark): De-duplicate the below so they don't get out of sync.

def update_grammar_with_typed_variables(grammar_dictionary: Dict[str, List[str]],
                                        prelinked_entities: Dict[str, Dict[str, str]],
                                        dataset_name: str) -> None:

    dataset_type_mapping = GLOBAL_DATASET_VARIABLE_TYPES[dataset_name]

    binary_ops = []

    for variable, _ in prelinked_entities.items():
        column_producers: Set[Tuple[str, str, str]] = dataset_type_mapping.get(variable, set())

        if not column_producers:
            print(f"Warning - {variable} not found in mapping.")
        for producer in column_producers:
            table, _, column = producer

            if not variable in GLOBAL_DATASET_VALUES.get(dataset_name, []):
                binary_ops.append(f'("{table}" ws "." ws "{column}" wsp binaryop wsp  "\'{variable}\'")')
                if dataset_name == "atis":
                    binary_ops.append(f'( ws "(" ws "{table}" ws "." ws "{column}" wsp binaryop wsp  "\'{variable}\'" ws ")" ws)')
            else:
                binary_ops.append(f'("{table}" ws "." ws "{column}" wsp binaryop wsp  "{variable}")')
                if dataset_name == "atis":
                    binary_ops.append(f'( ws "(" ws "{table}" ws "." ws "{column}" wsp binaryop wsp  "{variable}" ws ")" ws )')
 
        # TODO update the signatures for binary, tertiary and in_exprs here.
        grammar_dictionary["value"] = [f'"\'{variable}\'"'] + grammar_dictionary["value"]

    grammar_dictionary["expr"] = sorted(binary_ops, reverse=True) + grammar_dictionary["expr"]

def update_grammar_with_linked_typed_variables(grammar_dictionary: Dict[str, List[str]], # pylint: disable=invalid-name
                                               prelinked_entities: Dict[str, Dict[str, str]],
                                               dataset_name: str) -> None:

    dataset_type_mapping = GLOBAL_DATASET_VARIABLE_TYPES[dataset_name]

    binary_ops = []
    values_to_add = set()
    terminal_values: Dict[str, List[str]] = defaultdict(list)
    for variable, _ in prelinked_entities.items():
        column_producers: Set[Tuple[str, str, str]] = dataset_type_mapping.get(variable, set())

        if not column_producers:
            print(f"Warning -{variable} not found in mapping.")
            values_to_add.add(f'"\'{variable}\'"')
        for producer in column_producers:
            table, _, column = producer

            binary_ops.append(f'("{table}" ws "." ws "{column}" wsp binaryop wsp {table}_{column}_value)')
            if not variable in GLOBAL_DATASET_VALUES.get(dataset_name, []):
                terminal_values[f"{table}_{column}_value"].append(f'"\'{variable}\'"')
            else:
                terminal_values[f"{table}_{column}_value"].append(f"{variable}")

            values_to_add.add(f'{table}_{column}_value')

        # TODO update the signatures for binary, tertiary and in_exprs here.
    grammar_dictionary["value"] = sorted(list(values_to_add), reverse=True)+ grammar_dictionary["value"]

    for nonterminal, values in terminal_values.items():
        grammar_dictionary[nonterminal] = values

    grammar_dictionary["expr"] = sorted(binary_ops, reverse=True) + grammar_dictionary["expr"]


def update_grammar_values_with_variables(grammar_dictionary: Dict[str, List[str]],
                                         prelinked_entities: Dict[str, Dict[str, str]]) -> None:

    for variable, _ in prelinked_entities.items():
        grammar_dictionary["value"] = [f'"\'{variable}\'"'] + grammar_dictionary["value"]
