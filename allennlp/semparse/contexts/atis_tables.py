from collections import defaultdict
from typing import List, Dict
import re

from parsimonious.expressions import Sequence, OneOf, Literal
from parsimonious.grammar import Grammar

TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100
HOURS_IN_DAY = 2400
AROUND_RANGE = 30

SQL_GRAMMAR_STR = """
    stmt                = query ws ";" ws
    query               = (ws "(" ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws ")" ws) /
                          (ws "SELECT" ws distinct ws select_results ws "FROM" ws table_refs ws where_clause ws)
                        
    select_results      = col_refs / agg

    agg                 = agg_func ws "(" ws col_ref ws ")"
    agg_func            = "MIN" / "min" / "MAX" / "max" / "COUNT" / "count"

    col_refs            = (col_ref ws "," ws col_refs) / (col_ref)
    table_refs          = (table_name ws "," ws table_refs) / (table_name)

    where_clause        = ("WHERE" ws "(" ws conditions ws ")" ws) / ("WHERE" ws conditions ws)
    
    conditions          = (condition ws conj ws conditions) / 
                          (condition ws conj ws "(" ws conditions ws ")") /
                          ("(" ws conditions ws ")" ws conj ws conditions) /
                          ("(" ws conditions ws ")") /
                          (not ws conditions ws ) /
                          condition
    condition           = in_clause / ternaryexpr / biexpr

    in_clause           = (ws col_ref ws "IN" ws query ws)

    biexpr              = ( col_ref ws binaryop ws value) / (value ws binaryop ws value) / ( col_ref ws "LIKE" ws string)
    binaryop            = "+" / "-" / "*" / "/" / "=" /
                          ">=" / "<=" / ">" / "<"  / "is" / "IS"

    ternaryexpr         = (col_ref ws not "BETWEEN" ws value ws and value ws) /
                          (col_ref ws "BETWEEN" ws value ws and value ws)

    value               = (not ws pos_value) / (pos_value)
    pos_value           = ("ALL" ws query) / ("ANY" ws query) / number / boolean / col_ref / string / agg_results / "NULL"

    agg_results         = (ws "("  ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws ")" ws) /
                          (ws "SELECT" ws distinct ws agg ws "FROM" ws table_name ws where_clause ws)

    boolean             = "true" / "false"

    ws                  = ~"\s*"i

    conj                = and / or
    and                 = "AND" ws
    or                  = "OR" ws
    not                 = ("NOT" ws ) / ("not" ws)
    asterisk            = "*"
    distinct            = ("DISTINCT") / ("")

    col_ref             =  ""
    table_ref           =  ""
    table_name          =  ""
    number              =  ""
    string              =  ""

"""


OPTIONAL = ['lparen', 'rparen', '(lparen ws)', '(ws rparen)', 'not', 'ws', 'asterisk']

class ConversationContext():
    """
    A ``ConversationContext`` represents the interaction in which an utterance occurs.
    It initializes the global actions that are valid for every interaction. For each utterance,
    local actions are added and are valid for future utterances in the same interaction.
    """
    def __init__(self, interaction: List[Dict[str, str]]):
        self.interaction = interaction
        self.base_sql_def = SQL_GRAMMAR_STR
        self.grammar = Grammar(SQL_GRAMMAR_STR)
        self.valid_actions = self.initialize_valid_actions()

    def initialize_valid_actions(self) -> Dict[str, List[str]]:
        """
        Initialize the conversation context with global actions, these are valid for all contexts
        """
        valid_actions: Dict[str, List[str]] = defaultdict(set)

        for key in self.grammar:
            rhs = self.grammar[key]
            if isinstance(rhs, Sequence):
                valid_actions[key].add(" ".join(rhs._unicode_members()))

            elif isinstance(rhs, OneOf):
                for option in rhs._unicode_members():
                    valid_actions[key].add(option)

            elif isinstance(rhs, Literal):
                if rhs.literal != "":
                    valid_actions[key].add("%s" % rhs.literal)
                else:
                    valid_actions[key] = set()

        for table in list(sorted(TABLES.keys(), reverse=True)):
            valid_actions['table_name'].add(table)
            for column in sorted(TABLES[table], reverse=True):
                valid_actions['col_ref'].add('("{}" ws "." ws "{}")'.format(table, column))

        valid_actions['col_ref'].add('asterisk')
        valid_action_strings = {key: sorted(value) for key, value in valid_actions.items()}
        return valid_action_strings


def get_times_from_utterance(utterance: str) -> List[str]:
    """
    Given an utterance, get the numbers that correspond to times and convert time
    for example: convert ``7pm`` to 1900
    """
    pm_times = [int(pm_str.rstrip('pm')) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR
                for pm_str in re.findall(r'\d+pm', utterance)]
    am_times = [int(am_str.rstrip('am')) * HOUR_TO_TWENTY_FOUR
                for am_str in re.findall(r"\d+", utterance)]
    oclock_times = [int(oclock_str.rstrip("o'clock")) * HOUR_TO_TWENTY_FOUR
                    for oclock_str in re.findall(r"\d+\so'clock", utterance)]
    oclock_times = oclock_times + [(oclock_time + TWELVE_TO_TWENTY_FOUR) % HOURS_IN_DAY \
                                   for oclock_time in oclock_times]
    times = am_times + pm_times + oclock_times
    if 'noon' in utterance:
        times.append(1200)

    around_times = []
    if "around" in utterance or "about" in utterance:
        for time in times:
            around_times.append((time + AROUND_RANGE) % HOURS_IN_DAY)
            around_times.append((time - HOUR_TO_TWENTY_FOUR + AROUND_RANGE) % HOURS_IN_DAY)

    times += around_times

    return [str(time) for time in times]


def get_nums_from_utterance(utterance: str) -> List[str]:
    """
    Given an utterance, find all the numbers that are in the action space.
    """
    nums = ['1', '0']
    nums.extend(re.findall(r'\d+', utterance))
    nums.extend([str(int(num_str.rstrip('pm')) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR)
                 for num_str in re.findall(r'\d+', utterance)])

    nums.extend(get_times_from_utterance(utterance))

    words = utterance.split(' ')
    for word in words:
        if word in MONTH_NUMBERS:
            nums.append(str(MONTH_NUMBERS[word]))
        if word in DAY_NUMBERS:
            nums.append(str(DAY_NUMBERS[word]))
        if word in MISC_TIME_TRIGGERS:
            nums.extend(MISC_TIME_TRIGGERS[word])

    for tens, digits in zip(words, words[1:]):
        day = ' '.join([tens, digits])
        if day in DAY_NUMBERS:
            nums.append(str(DAY_NUMBERS[day]))
    return sorted(nums, reverse=True)

AIRLINE_CODES = {'argentina': 'AR',
                 'alliance': '3J',
                 'canada': 'AC',
                 'ontario': 'GX',
                 'wisconson': 'ZW',
                 'alaska': 'AS',
                 'alpha': '7V',
                 'american': 'AA',
                 'american trans': 'TZ',
                 'america west': 'HP',
                 'atlantic': 'DH',
                 'atlantic.': 'EV',
                 'braniff.': 'BE',
                 'british': 'BA',
                 'business': 'HQ',
                 'canadian': 'CP',
                 'carnival': 'KW',
                 'christman': 'SX',
                 'colgan': '9L',
                 'comair': 'OH',
                 'continental': 'CO',
                 'czecho': 'OK',
                 'delta': 'DL',
                 'express': '9E',
                 'eastern': 'EA',
                 'grand': 'QD',
                 'lufthansa': 'LH',
                 'mesaba': 'XJ',
                 'mgm': 'MG',
                 'midwest': 'YX',
                 'nation': 'NX',
                 'northeast': '2V',
                 'northwest': 'NW',
                 'ontario express': '9X',
                 'precision': 'RP',
                 'royal': 'AT',
                 'sabena': 'SN',
                 'sky': 'OO',
                 'south': 'WN',
                 'thai': 'TG',
                 'tower': 'FF',
                 'states': '9N',
                 'twa': 'TW',
                 'world': 'RZ',
                 'united': 'UA',
                 'us': 'US',
                 'west': 'OE'}

CITY_CODES = {'ATLANTA': 'MATL',
              'BALTIMORE': 'BBWI',
              'BOSTON': 'BBOS',
              'BURBANK': 'BBUR',
              'CHARLOTTE': 'CCLT',
              'CHICAGO': 'CCHI',
              'CINCINNATI': 'CCVG',
              'CLEVELAND': 'CCLE',
              'COLUMBUS': 'CCMH',
              'DALLAS': 'DDFW',
              'DENVER': 'DDEN',
              'DETROIT': 'DDTT',
              'FORT WORTH': 'FDFW',
              'HOUSTON': 'HHOU',
              'INDIANAPOLIS': 'IIND',
              'KANSAS CITY': 'MMKC',
              'LAS VEGAS': 'LLAS',
              'LONG BEACH': 'LLGB',
              'LOS ANGELES': 'LLAX',
              'MEMPHIS': 'MMEM',
              'MIAMI': 'MMIA',
              'MILWAUKEE': 'MMKE',
              'MINNEAPOLIS': 'MMSP',
              'MONTREAL': 'YYMQ',
              'NASHVILLE': 'BBNA',
              'NEW YORK': 'NNYC',
              'NEWARK': 'JNYC',
              'OAKLAND': 'OOAK',
              'ONTARIO': 'OONT',
              'ORLANDO': 'OORL',
              'PHILADELPHIA': 'PPHL',
              'PHOENIX': 'PPHX',
              'PITTSBURGH': 'PPIT',
              'SALT LAKE CITY': 'SSLC',
              'SAN DIEGO': 'SSAN',
              'SAN FRANCISCO': 'SSFO',
              'SAN JOSE': 'SSJC',
              'SEATTLE': 'SSEA',
              'ST. LOUIS': 'SSTL',
              'ST. PAUL': 'SMSP',
              'ST. PETERSBURG': 'STPA',
              'TACOMA': 'TSEA',
              'TAMPA': 'TTPA',
              'TORONTO': 'YYTO',
              'WASHINGTON': 'WWAS',
              'WESTCHESTER COUNTY': 'HHPN'}




MONTH_NUMBERS = {
        'january': 1,
        'february': 2,
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8,
        'september': 9,
        'october': 10,
        'november': 11,
        'december': 12,
        }

DAY_NUMBERS = {
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'ninth': 9,
        'tenth': 10,
        'eleventh': 11,
        'twelfth': 12,
        'thirteenth': 13,
        'fourteenth': 14,
        'fifteenth': 15,
        'sixteenth': 16,
        'seventeenth': 17,
        'eighteenth': 18,
        'nineteenth': 19,
        'twentieth': 20,
        'twenty first': 21,
        'twenty second': 22,
        'twenty third': 23,
        'twenty fourth': 24,
        'twenty fifth': 25,
        'twenty sixth': 26,
        'twenty seventh': 27,
        'twenty eighth': 28,
        'twenty ninth': 29,
        'thirtieth': 30,
        'thirty first': 31}

GROUND_SERVICE = {
        'air taxi': 'AIR TAXI OPERATION',
        'limo': 'LIMOUSINE',
        'rapid': 'RAPID TRANSIT',
        'rental': 'RENTAL CAR',
        'car': 'RENTAL CAR',
        'taxi': 'TAXI'}

MISC_STR = {"every day" : "DAILY"}

MISC_TIME_TRIGGERS = {'morning': ['0', '1200'],
                      'afternoon': ['1200', '1800'],
                      'after': ['1200', '1800'],
                      'evening': ['1800', '2200'],
                      'late evening': ['2000', '2200'],
                      'lunch': ['1400'],
                      'noon': ['1200']}


TABLES = {'aircraft': ['aircraft_code', 'aircraft_description',
                       'manufacturer', 'basic_type', 'propulsion',
                       'wide_body', 'pressurized'],
          'airline': ['airline_name', 'airline_code'],
          'airport': ['airport_code', 'airport_name', 'airport_location',
                      'state_code', 'country_name', 'time_zone_code',
                      'minimum_connect_time'],
          'airport_service': ['city_code', 'airport_code', 'miles_distant',
                              'direction', 'minutes_distant'],
          'city': ['city_code', 'city_name', 'state_code', 'country_name', 'time_zone_code'],
          'class_of_service': ['booking_class', 'rank', 'class_description'],
          'date_day': ['month_number', 'day_number', 'year', 'day_name'],
          'days': ['days_code', 'day_name'],
          'equipment_sequence': ['aircraft_code_sequence', 'aircraft_code'],
          'fare': ['fare_id', 'from_airport', 'to_airport', 'fare_basis_code', 'fare_airline',
                   'restriction_code', 'one_direction_cost', 'round_trip_cost', 'round_trip_required'],
          'fare_basis': ['fare_basis_code', 'booking_class', 'class_type', 'premium', 'economy',
                         'discounted', 'night', 'season', 'basis_days'],
          'flight': ['flight_id', 'flight_days', 'from_airport', 'to_airport', 'departure_time',
                     'arrival_time', 'airline_flight', 'airline_code', 'flight_number',
                     'aircraft_code_sequence', 'meal_code', 'stops', 'connections',
                     'dual_carrier', 'time_elapsed'],
          'flight_fare': ['flight_id', 'fare_id'],
          'flight_leg': ['flight_id', 'leg_number', 'leg_flight'],
          'flight_stop': ['flight_id', 'stop_number', 'stop_days', 'stop_airport', 'arrival_time',
                          'arrival_airline', 'arrival_flight_number', 'departure_time', 'departure_airline',
                          'departure_flight_number', 'stop_time'],
          'food_service': ['meal_code', 'meal_number', 'compartment', 'meal_description'],
          'ground_service': ['city_code', 'airport_code', 'transport_type', 'ground_fare'],
          'month': ['month_number', 'month_name'],
          'restriction': ['restriction_code', 'advance_purchase', 'stopovers', 'saturday_stay_required',
                          'minimum_stay', 'maximum_stay', 'application', 'no_discounts'],
          'state': ['state_code', 'state_name', 'country_name']}



YES_NO = {'one way': 'NO',
          'economy': 'YES'}


AIRPORT_CODES = ['ATL', 'NA', 'OS', 'UR', 'WI', 'CLE', 'CLT', 'CMH',
                 'CVG', 'DAL', 'DCA', 'DEN', 'DET', 'DFW', 'DTW',
                 'EWR', 'HOU', 'HPN', 'IAD', 'IAH', 'IND', 'JFK',
                 'LAS', 'LAX', 'LGA', 'LG', 'MCI', 'MCO', 'MDW', 'MEM',
                 'MIA', 'MKE', 'MSP', 'OAK', 'ONT', 'ORD', 'PHL', 'PHX',
                 'PIE', 'PIT', 'SAN', 'SEA', 'SFO', 'SJC', 'SLC',
                 'STL', 'TPA', 'YKZ', 'YMX', 'YTZ', 'YUL', 'YYZ']

AIRLINE_CODE_LIST = ['AR', '3J', 'AC', '9X', 'ZW', 'AS', '7V',
                     'AA', 'TZ', 'HP', 'DH', 'EV', 'BE', 'BA',
                     'HQ', 'CP', 'KW', 'SX', '9L', 'OH', 'CO',
                     'OK', 'DL', '9E', 'QD', 'LH', 'XJ', 'MG',
                     'YX', 'NX', '2V', 'NW', 'RP', 'AT', 'SN',
                     'OO', 'WN', 'TG', 'FF', '9N', 'TW', 'RZ',
                     'UA', 'US', 'OE']

CITIES = ['NASHVILLE', 'BOSTON', 'BURBANK', 'BALTIMORE', 'CHICAGO', 'CLEVELAND',
          'CHARLOTTE', 'COLUMBUS', 'CINCINNATI', 'DENVER', 'DALLAS', 'DETROIT',
          'FORT WORTH', 'HOUSTON', 'WESTCHESTER COUNTY', 'INDIANAPOLIS', 'NEWARK',
          'LAS VEGAS', 'LOS ANGELES', 'LONG BEACH', 'ATLANTA', 'MEMPHIS', 'MIAMI',
          'KANSAS CITY', 'MILWAUKEE', 'MINNEAPOLIS', 'NEW YORK', 'OAKLAND', 'ONTARIO',
          'ORLANDO', 'PHILADELPHIA', 'PHOENIX', 'PITTSBURGH', 'ST. PAUL', 'SAN DIEGO',
          'SEATTLE', 'SAN FRANCISCO', 'SAN JOSE', 'SALT LAKE CITY', 'ST. LOUIS',
          'ST. PETERSBURG', 'TACOMA', 'TAMPA', 'WASHINGTON', 'MONTREAL', 'TORONTO']

CITY_CODE_LIST = ['BBNA', 'BBOS', 'BBUR', 'BBWI', 'CCHI', 'CCLE', 'CCLT', 'CCMH', 'CCVG', 'DDEN',
                  'DDFW', 'DDTT', 'FDFW', 'HHOU', 'HHPN', 'IIND', 'JNYC', 'LLAS', 'LLAX', 'LLGB',
                  'MATL', 'MMEM', 'MMIA', 'MMKC', 'MMKE', 'MMSP', 'NNYC', 'OOAK', 'OONT', 'OORL',
                  'PPHL', 'PPHX', 'PPIT', 'SMSP', 'SSAN', 'SSEA', 'SSFO', 'SSJC', 'SSLC', 'SSTL',
                  'STPA', 'TSEA', 'TTPA', 'WWAS', 'YYMQ', 'YYTO']

CLASS = ['COACH', 'BUSINESS', 'FIRST', 'THRIST', 'STANDARD', 'SHUTTLE']

DAY_OF_WEEK = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']

FARE_BASIS_CODE = ['B', 'BH', 'BHW', 'BHX', 'BL', 'BLW', 'BLX', 'BN', 'BOW', 'BOX',
                   'BW', 'BX', 'C', 'CN', 'F', 'FN', 'H', 'HH', 'HHW', 'HHX', 'HL', 'HLW', 'HLX',
                   'HOW', 'HOX', 'J', 'K', 'KH', 'KL', 'KN', 'LX', 'M', 'MH', 'ML', 'MOW', 'P',
                   'Q', 'QH', 'QHW', 'QHX', 'QLW', 'QLX', 'QO', 'QOW', 'QOX', 'QW', 'QX', 'S',
                   'U', 'V', 'VHW', 'VHX', 'VW', 'VX', 'Y', 'YH', 'YL', 'YN', 'YW', 'YX']

MEALS = ['BREAKFAST', 'LUNCH', 'SNACK', 'DINNER']

RESTRICT_CODES = ['AP/2', 'AP/6', 'AP/12', 'AP/20', 'AP/21', 'AP/57', 'AP/58', 'AP/60',
                  'AP/75', 'EX/9', 'EX/13', 'EX/14', 'EX/17', 'EX/19']

STATES = ['ARIZONA', 'CALIFORNIA', 'COLORADO', 'DISTRICT OF COLUMBIA',
          'FLORIDA', 'GEORGIA', 'ILLINOIS', 'INDIANA', 'MASSACHUSETTS',
          'MARYLAND', 'MICHIGAN', 'MINNESOTA', 'MISSOURI', 'NORTH CAROLINA',
          'NEW JERSEY', 'NEVADA', 'NEW YORK', 'OHIO', 'ONTARIO', 'PENNSYLVANIA',
          'QUEBEC', 'TENNESSEE', 'TEXAS', 'UTAH', 'WASHINGTON', 'WISCONSIN']

STATE_CODES = ['TN', 'MA', 'CA', 'MD', 'IL', 'OH', 'NC', 'CO', 'TX', 'MI', 'NY',
               'IN', 'NJ', 'NV', 'GA', 'FL', 'MO', 'WI', 'MN', 'PA', 'AZ', 'WA',
               'UT', 'DC', 'PQ', 'ON']
