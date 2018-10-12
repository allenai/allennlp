from typing import List, Dict, Callable, Set
from datetime import datetime, timedelta
import re
from collections import defaultdict
from nltk import ngrams

from allennlp.data.tokenizers import Token

TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100
HOURS_IN_DAY = 2400
AROUND_RANGE = 30
MINS_IN_HOUR = 60

APPROX_WORDS = ['about', 'around', 'approximately']
WORDS_PRECEDING_TIME = ['at', 'between', 'to', 'before', 'after']


def pm_map_match_to_query_value(match: str):
    if len(match.rstrip('pm')) < 3: # This will match something like ``5pm``.
        if match.startswith('12'):
            return [int(match.rstrip('pm')) * HOUR_TO_TWENTY_FOUR]
        else:
            return [int(match.rstrip('pm')) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR]
    else: # This will match something like ``530pm``.
        if match.startswith('12'):
            return [int(match.rstrip('pm'))]
        else:
            return [int(match.rstrip('pm')) + TWELVE_TO_TWENTY_FOUR]

def am_map_match_to_query_value(match: str):
    if len(match.rstrip('am')) < 3:
        return [int(match.rstrip('am')) * HOUR_TO_TWENTY_FOUR]
    else:
        return [int(match.rstrip('am'))]

def get_times_from_utterance(utterance: str,
                             char_offset_to_token_index: Dict[int, int],
                             indices_of_approximate_words: Set[int]) -> Dict[str, List[int]]:
    """
    Given an utterance, we get the numbers that correspond to times and convert them to
    values that may appear in the query. For example: convert ``7pm`` to ``1900``.
    """

    pm_linking_dict = _time_regex_match(r'\d+pm',
                                        utterance,
                                        char_offset_to_token_index,
                                        pm_map_match_to_query_value,
                                        indices_of_approximate_words)

    am_linking_dict = _time_regex_match(r'\d+am',
                                        utterance,
                                        char_offset_to_token_index,
                                        am_map_match_to_query_value,
                                        indices_of_approximate_words)

    oclock_linking_dict = _time_regex_match(r"\d+ o'clock",
                                            utterance,
                                            char_offset_to_token_index,
                                            lambda match: digit_to_query_time(match.rstrip(" o'clock")),
                                            indices_of_approximate_words)

    hours_linking_dict = _time_regex_match(r"\d+ hours",
                                           utterance,
                                           char_offset_to_token_index,
                                           lambda match: [int(match.rstrip(" hours"))],
                                           indices_of_approximate_words)


    times_linking_dict: Dict[str, List[int]] = defaultdict(list)
    linking_dicts = [pm_linking_dict, am_linking_dict, oclock_linking_dict, hours_linking_dict]

    for linking_dict in linking_dicts:
        for key, value in linking_dict.items():
            times_linking_dict[key].extend(value)

    return times_linking_dict

def get_date_from_utterance(tokenized_utterance: List[Token],
                            year: int = 1993) -> List[datetime]:
    """
    When the year is not explicitly mentioned in the utterance, the query assumes that
    it is 1993 so we do the same here. If there is no mention of the month or day then
    we do not return any dates from the utterance.
    """

    dates = []

    utterance = ' '.join([token.text for token in tokenized_utterance])
    year_result = re.findall(r'199[0-4]', utterance)
    if year_result:
        year = int(year_result[0])
    trigrams = ngrams([token.text for token in tokenized_utterance], 3)
    for month, tens, digit in trigrams:
        # This will match something like ``september twenty first``.
        day = ' '.join([tens, digit])
        if month in MONTH_NUMBERS and day in DAY_NUMBERS:
            try:
                dates.append(datetime(year, MONTH_NUMBERS[month], DAY_NUMBERS[day]))
            except ValueError:
                print('invalid month day')

    bigrams = ngrams([token.text for token in tokenized_utterance], 2)
    for month, day in bigrams:
        if month in MONTH_NUMBERS and day in DAY_NUMBERS:
            # This will match something like ``september first``.
            try:
                dates.append(datetime(year, MONTH_NUMBERS[month], DAY_NUMBERS[day]))
            except ValueError:
                print('invalid month day')

    fivegrams = ngrams([token.text for token in tokenized_utterance], 5)
    for tens, digit, _, year_match, month in fivegrams:
        # This will match something like ``twenty first of 1993 july``.
        day = ' '.join([tens, digit])
        if month in MONTH_NUMBERS and day in DAY_NUMBERS and year_match.isdigit():
            try:
                dates.append(datetime(int(year_match), MONTH_NUMBERS[month], DAY_NUMBERS[day]))
            except ValueError:
                print('invalid month day')
        if month in MONTH_NUMBERS and digit in DAY_NUMBERS and year_match.isdigit():
            try:
                dates.append(datetime(int(year_match), MONTH_NUMBERS[month], DAY_NUMBERS[digit]))
            except ValueError:
                print('invalid month day')
    return dates

def get_numbers_from_utterance(utterance: str, tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    """
    Given an utterance, this function finds all the numbers that are in the action space. Since we need to
    keep track of linking scores, we represent the numbers as a dictionary, where the keys are the string
    representation of the number and the values are lists of the token indices that triggers that number.
    """
    # When we use a regex to find numbers or strings, we need a mapping from
    # the character to which token triggered it.
    char_offset_to_token_index = {token.idx : token_index
                                  for token_index, token in enumerate(tokenized_utterance)}

    # We want to look up later for each time whether it appears after a word
    # such as "about" or "approximately".
    indices_of_approximate_words = {index for index, token in enumerate(tokenized_utterance)
                                    if token.text in APPROX_WORDS}

    indices_of_words_preceding_time = {index for index, token in enumerate(tokenized_utterance)
                                       if token.text in WORDS_PRECEDING_TIME}

    indices_of_am_pm = {index for index, token in enumerate(tokenized_utterance)
                        if token.text in {'am', 'pm'}}

    number_linking_dict: Dict[str, List[int]] = defaultdict(list)

    for token_index, token in enumerate(tokenized_utterance):
        if token.text.isdigit():
            if token_index - 1 in indices_of_words_preceding_time and token_index + 1 not in indices_of_am_pm:
                for time in digit_to_query_time(token.text):
                    number_linking_dict[str(time)].append(token_index)
    times_linking_dict = get_times_from_utterance(utterance,
                                                  char_offset_to_token_index,
                                                  indices_of_approximate_words)
    for key, value in times_linking_dict.items():
        number_linking_dict[key].extend(value)

    for index, token in enumerate(tokenized_utterance):
        for number in NUMBER_TRIGGER_DICT.get(token.text, []):
            if index - 1 in indices_of_approximate_words:
                for approx_time in get_approximate_times([int(number)]):
                    number_linking_dict[str(approx_time)].append(index)
            else:
                number_linking_dict[number].append(index)
    return number_linking_dict

def get_time_range_start_from_utterance(utterance: str, # pylint: disable=unused-argument
                                        tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    late_indices = {index for index, token in enumerate(tokenized_utterance)
                    if token.text == 'late'}

    time_range_start_linking_dict: Dict[str, List[int]] = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        for time in TIME_RANGE_START_DICT.get(token.text, []):
            if token_index - 1 not in late_indices:
                time_range_start_linking_dict[str(time)].append(token_index)

    bigrams = ngrams([token.text for token in tokenized_utterance], 2)
    for bigram_index, bigram in enumerate(bigrams):
        for time in TIME_RANGE_START_DICT.get(' '.join(bigram), []):
            time_range_start_linking_dict[str(time)].extend([bigram_index, bigram_index + 1])

    return time_range_start_linking_dict

def get_time_range_end_from_utterance(utterance: str, # pylint: disable=unused-argument
                                      tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    early_indices = {index for index, token in enumerate(tokenized_utterance)
                     if token.text == 'early'}

    time_range_end_linking_dict: Dict[str, List[int]] = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        for time in TIME_RANGE_END_DICT.get(token.text, []):
            if token_index - 1 not in early_indices:
                time_range_end_linking_dict[str(time)].append(token_index)

    bigrams = ngrams([token.text for token in tokenized_utterance], 2)
    for bigram_index, bigram in enumerate(bigrams):
        for time in TIME_RANGE_END_DICT.get(' '.join(bigram), []):
            time_range_end_linking_dict[str(time)].extend([bigram_index, bigram_index + 1])

    return time_range_end_linking_dict

def get_costs_from_utterance(utterance: str, # pylint: disable=unused-argument
                             tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    dollars_indices = {index for index, token in enumerate(tokenized_utterance)
                       if token.text == 'dollars' or token.text == 'dollar'}

    costs_linking_dict: Dict[str, List[int]] = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        if token_index + 1 in dollars_indices and token.text.isdigit():
            costs_linking_dict[token.text].append(token_index)
    return costs_linking_dict

def get_flight_numbers_from_utterance(utterance: str, # pylint: disable=unused-argument
                                      tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    indices_words_preceding_flight_number = {index for index, token in enumerate(tokenized_utterance)
                                             if token.text in {'flight', 'number'}
                                             or token.text.upper() in AIRLINE_CODE_LIST
                                             or token.text.lower() in AIRLINE_CODES.keys()}

    indices_words_succeeding_flight_number = {index for index, token in enumerate(tokenized_utterance)
                                              if token.text == 'flight'}

    flight_numbers_linking_dict: Dict[str, List[int]] = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        if token.text.isdigit():
            if token_index - 1 in indices_words_preceding_flight_number:
                flight_numbers_linking_dict[token.text].append(token_index)
            if token_index + 1 in indices_words_succeeding_flight_number:
                flight_numbers_linking_dict[token.text].append(token_index)
    return flight_numbers_linking_dict

def digit_to_query_time(digit: str) -> List[int]:
    """
    Given a digit in the utterance, return a list of the times that it corresponds to.
    """
    if len(digit) > 2:
        return [int(digit), int(digit) + TWELVE_TO_TWENTY_FOUR]
    elif int(digit) % 12 == 0:
        return [0, 1200, 2400]
    return [int(digit) * HOUR_TO_TWENTY_FOUR,
            (int(digit) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR) % HOURS_IN_DAY]

def get_approximate_times(times: List[int]) -> List[int]:
    """
    Given a list of times that follow a word such as ``about``,
    we return a list of times that could appear in the query as a result
    of this. For example if ``about 7pm`` appears in the utterance, then
    we also want to add ``1830`` and ``1930``.
    """
    approximate_times = []
    for time in times:
        hour = int(time/HOUR_TO_TWENTY_FOUR) % 24
        minute = time % HOUR_TO_TWENTY_FOUR
        approximate_time = datetime.now()
        approximate_time = approximate_time.replace(hour=hour, minute=minute)

        start_time_range = approximate_time - timedelta(minutes=30)
        end_time_range = approximate_time + timedelta(minutes=30)
        approximate_times.extend([start_time_range.hour * HOUR_TO_TWENTY_FOUR + start_time_range.minute,
                                  end_time_range.hour * HOUR_TO_TWENTY_FOUR + end_time_range.minute])

    return approximate_times

def _time_regex_match(regex: str,
                      utterance: str,
                      char_offset_to_token_index: Dict[int, int],
                      map_match_to_query_value: Callable[[str], List[int]],
                      indices_of_approximate_words: Set[int]) -> Dict[str, List[int]]:
    r"""
    Given a regex for matching times in the utterance, we want to convert the matches
    to the values that appear in the query and token indices they correspond to.

    ``char_offset_to_token_index`` is a dictionary that maps from the character offset to
    the token index, we use this to look up what token a regex match corresponds to.
    ``indices_of_approximate_words`` are the token indices of the words such as ``about`` or
    ``approximately``. We use this to check if a regex match is preceded by one of these words.
    If it is, we also want to add the times that define this approximate time range.

    ``map_match_to_query_value`` is a function that converts the regex matches to the
    values that appear in the query. For example, we may pass in a regex such as ``\d+pm``
    that matches times such as ``7pm``. ``map_match_to_query_value`` would be a function that
    takes ``7pm`` as input and returns ``1900``.
    """
    linking_scores_dict: Dict[str, List[int]] = defaultdict(list)
    number_regex = re.compile(regex)
    for match in number_regex.finditer(utterance):
        query_values = map_match_to_query_value(match.group())
        # If the time appears after a word like ``about`` then we also add
        # the times that mark the start and end of the allowed range.
        approximate_times = []
        if char_offset_to_token_index.get(match.start(), 0) - 1 in indices_of_approximate_words:
            approximate_times.extend(get_approximate_times(query_values))
        query_values.extend(approximate_times)
        if match.start() in char_offset_to_token_index:
            for query_value in query_values:
                linking_scores_dict[str(query_value)].extend([char_offset_to_token_index[match.start()],
                                                              char_offset_to_token_index[match.start()] + 1])
    return linking_scores_dict

def get_trigger_dict(trigger_lists: List[List[str]],
                     trigger_dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged_trigger_dict: Dict[str, List[str]] = defaultdict(list)
    for trigger_list in trigger_lists:
        for trigger in trigger_list:
            merged_trigger_dict[trigger.lower()].append(trigger)

    for trigger_dict in trigger_dicts:
        for key, value in trigger_dict.items():
            merged_trigger_dict[key.lower()].extend(value)

    return merged_trigger_dict

def convert_to_string_list_value_dict(trigger_dict: Dict[str, int]) -> Dict[str, List[str]]:
    return {key: [str(value)] for key, value in trigger_dict.items()}

AIRLINE_CODES = {'alaska': ['AS'],
                 'alliance': ['3J'],
                 'alpha': ['7V'],
                 'america west': ['HP'],
                 'american': ['AA'],
                 'american trans': ['TZ'],
                 'argentina': ['AR'],
                 'atlantic': ['DH'],
                 'atlantic.': ['EV'],
                 'braniff.': ['BE'],
                 'british': ['BA'],
                 'business': ['HQ'],
                 'canada': ['AC'],
                 'canadian': ['CP'],
                 'carnival': ['KW'],
                 'christman': ['SX'],
                 'colgan': ['9L'],
                 'comair': ['OH'],
                 'continental': ['CO'],
                 'czecho': ['OK'],
                 'delta': ['DL'],
                 'eastern': ['EA'],
                 'express': ['9E'],
                 'grand': ['QD'],
                 'lufthansa': ['LH'],
                 'mesaba': ['XJ'],
                 'mgm': ['MG'],
                 'midwest': ['YX'],
                 'nation': ['NX'],
                 'nationair': ['NX'],
                 'northeast': ['2V'],
                 'northwest': ['NW'],
                 'ontario': ['GX'],
                 'ontario express': ['9X'],
                 'precision': ['RP'],
                 'royal': ['AT'],
                 'sabena': ['SN'],
                 'sky': ['OO'],
                 'south': ['WN'],
                 'states': ['9N'],
                 'thai': ['TG'],
                 'tower': ['FF'],
                 'twa': ['TW'],
                 'united': ['UA'],
                 'us': ['US'],
                 'west': ['OE'],
                 'wisconson': ['ZW'],
                 'world': ['RZ']}

CITY_CODES = {'ATLANTA': ['MATL'],
              'BALTIMORE': ['BBWI'],
              'BOSTON': ['BBOS'],
              'BURBANK': ['BBUR'],
              'CHARLOTTE': ['CCLT'],
              'CHICAGO': ['CCHI'],
              'CINCINNATI': ['CCVG'],
              'CLEVELAND': ['CCLE'],
              'COLUMBUS': ['CCMH'],
              'DALLAS': ['DDFW'],
              'DENVER': ['DDEN'],
              'DETROIT': ['DDTT'],
              'FORT WORTH': ['FDFW'],
              'HOUSTON': ['HHOU'],
              'KANSAS CITY': ['MMKC'],
              'LAS VEGAS': ['LLAS'],
              'LONG BEACH': ['LLGB'],
              'LOS ANGELES': ['LLAX'],
              'MEMPHIS': ['MMEM'],
              'MIAMI': ['MMIA'],
              'MILWAUKEE': ['MMKE'],
              'MINNEAPOLIS': ['MMSP'],
              'MONTREAL': ['YYMQ'],
              'NASHVILLE': ['BBNA'],
              'NEW YORK': ['NNYC'],
              'NEWARK': ['JNYC'],
              'OAKLAND': ['OOAK'],
              'ONTARIO': ['OONT'],
              'ORLANDO': ['OORL'],
              'PHILADELPHIA': ['PPHL'],
              'PHOENIX': ['PPHX'],
              'PITTSBURGH': ['PPIT'],
              'SALT LAKE CITY': ['SSLC'],
              'SAN DIEGO': ['SSAN'],
              'SAN FRANCISCO': ['SSFO'],
              'SAN JOSE': ['SSJC'],
              'SEATTLE': ['SSEA'],
              'ST. LOUIS': ['SSTL'],
              'ST. PAUL': ['SMSP'],
              'ST. PETERSBURG': ['STPA'],
              'TACOMA': ['TSEA'],
              'TAMPA': ['TTPA'],
              'TORONTO': ['YYTO'],
              'WASHINGTON': ['WWAS'],
              'WESTCHESTER COUNTY': ['HHPN']}

MONTH_NUMBERS = {'january': 1,
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
                 'december': 12}

GROUND_SERVICE = {'air taxi': ['AIR TAXI OPERATION'],
                  'car': ['RENTAL CAR'],
                  'limo': ['LIMOUSINE'],
                  'limousine': ['LIMOUSINE'],
                  'rapid': ['RAPID TRANSIT'],
                  'rental': ['RENTAL CAR'],
                  'taxi': ['TAXI']}

MISC_STR = {"every day" : ["DAILY"],
            "saint petersburg": ["ST. PETERSBURG"],
            "saint louis": ["ST. LOUIS"]}

DAY_NUMBERS = {'first': 1,
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


MISC_TIME_TRIGGERS = {'lunch': ['1400'],
                      'noon': ['1200'],
                      'early evening': ['1800', '2000'],
                      'morning': ['0', '1200'],
                      'night': ['1800', '2400']}

TIME_RANGE_START_DICT = {'morning': ['0'],
                         'mornings': ['1200'],
                         'afternoon': ['1200'],
                         'afternoons': ['1200'],
                         'after noon': ['1200'],
                         'late afternoon': ['1600'],
                         'evening': ['1800'],
                         'late evening': ['2000']}

TIME_RANGE_END_DICT = {'early morning': ['800'],
                       'morning': ['1200', '800'],
                       'mornings': ['1200', '800'],
                       'early afternoon': ['1400'],
                       'afternoon': ['1800'],
                       'afternoons': ['1800'],
                       'after noon': ['1800'],
                       'evening': ['2200']}

ALL_TABLES = {'aircraft': ['aircraft_code', 'aircraft_description', 'capacity',
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
              'date_day': ['day_name'],
              'days': ['days_code', 'day_name'],
              'equipment_sequence': ['aircraft_code_sequence', 'aircraft_code'],
              'fare': ['fare_id', 'from_airport', 'to_airport', 'fare_basis_code',
                       'fare_airline', 'restriction_code', 'one_direction_cost',
                       'round_trip_cost', 'round_trip_required'],
              'fare_basis': ['fare_basis_code', 'booking_class', 'class_type', 'premium', 'economy',
                             'discounted', 'night', 'season', 'basis_days'],
              'flight': ['flight_id', 'flight_days', 'from_airport', 'to_airport', 'departure_time',
                         'arrival_time', 'airline_flight', 'airline_code', 'flight_number',
                         'aircraft_code_sequence', 'meal_code', 'stops', 'connections',
                         'dual_carrier', 'time_elapsed'],
              'flight_fare': ['flight_id', 'fare_id'],
              'flight_leg': ['flight_id', 'leg_number', 'leg_flight'],
              'flight_stop': ['flight_id', 'stop_number', 'stop_days', 'stop_airport',
                              'arrival_time', 'arrival_airline', 'arrival_flight_number',
                              'departure_time', 'departure_airline', 'departure_flight_number',
                              'stop_time'],
              'food_service': ['meal_code', 'meal_number', 'compartment', 'meal_description'],
              'ground_service': ['city_code', 'airport_code', 'transport_type', 'ground_fare'],
              'month': ['month_number', 'month_name'],
              'restriction': ['restriction_code', 'advance_purchase', 'stopovers',
                              'saturday_stay_required', 'minimum_stay', 'maximum_stay',
                              'application', 'no_discounts'],
              'state': ['state_code', 'state_name', 'country_name']}

TABLES_WITH_STRINGS = {'airline' : ['airline_code', 'airline_name'],
                       'city' : ['city_name', 'state_code', 'city_code'],
                       'fare' : ['round_trip_required', 'fare_basis_code', 'restriction_code'],
                       'flight' : ['airline_code', 'flight_days'],
                       'flight_stop' : ['stop_airport'],
                       'airport' : ['airport_code', 'airport_name'],
                       'state' : ['state_name', 'state_code'],
                       'fare_basis' : ['fare_basis_code', 'class_type', 'economy', 'booking_class'],
                       'class_of_service' : ['booking_class', 'class_description'],
                       'aircraft' : ['basic_type', 'manufacturer', 'aircraft_code', 'propulsion'],
                       'restriction' : ['restriction_code'],
                       'ground_service' : ['transport_type'],
                       'days' : ['day_name', 'days_code'],
                       'food_service': ['meal_description', 'compartment']}

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

DAY_OF_WEEK_DICT = {'weekdays' : ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY']}
YES_NO = {'one way': ['NO'],
          'economy': ['YES']}
CITY_AIRPORT_CODES = {'atlanta' : ['ATL'],
                      'boston' : ['BOS'],
                      'baltimore': ['BWI'],
                      'charlotte': ['CLT'],
                      'dallas': ['DFW'],
                      'detroit': ['DTW'],
                      'houston': ['IAH'],
                      'la guardia': ['LGA'],
                      'love field': ['DAL'],
                      'los angeles': ['LAX'],
                      'oakland': ['OAK'],
                      'philadelphia': ['PHL'],
                      'pittsburgh': ['PIT'],
                      'san francisco': ['SFO'],
                      'toronto': ['YYZ']}
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
                     'UA', 'US', 'OE', 'EA']
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

CLASS = ['COACH', 'BUSINESS', 'FIRST', 'THRIFT', 'STANDARD', 'SHUTTLE']

AIRCRAFT_MANUFACTURERS = ['BOEING', 'MCDONNELL DOUGLAS', 'FOKKER']

AIRCRAFT_BASIC_CODES = ['DC9', '737', '767', '747', 'DC10', '757', 'MD80']

DAY_OF_WEEK_INDEX = {idx : [day] for idx, day in enumerate(DAY_OF_WEEK)}

TRIGGER_LISTS = [CITIES, AIRPORT_CODES,
                 STATES, STATE_CODES,
                 FARE_BASIS_CODE, CLASS,
                 AIRLINE_CODE_LIST, DAY_OF_WEEK,
                 CITY_CODE_LIST, MEALS,
                 RESTRICT_CODES,
                 AIRCRAFT_MANUFACTURERS,
                 AIRCRAFT_BASIC_CODES]

TRIGGER_DICTS = [CITY_AIRPORT_CODES,
                 AIRLINE_CODES,
                 CITY_CODES,
                 GROUND_SERVICE,
                 DAY_OF_WEEK_DICT,
                 YES_NO,
                 MISC_STR]
ATIS_TRIGGER_DICT = get_trigger_dict(TRIGGER_LISTS, TRIGGER_DICTS)

NUMBER_TRIGGER_DICT: Dict[str, List[str]] = get_trigger_dict([], [MISC_TIME_TRIGGERS])
