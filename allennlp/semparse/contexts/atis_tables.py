from typing import List, Dict, Callable, Set
from datetime import datetime
import re

from collections import defaultdict
from allennlp.data.tokenizers import Token

TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100
HOURS_IN_DAY = 2400
AROUND_RANGE = 30

APPROX_WORDS = ['about', 'around', 'approximately']
WORDS_PRECEDING_TIME = ['at', 'between', 'to', 'before', 'after']

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
                                        lambda match: [int(match.rstrip('pm'))
                                                       * HOUR_TO_TWENTY_FOUR +
                                                       TWELVE_TO_TWENTY_FOUR],
                                        indices_of_approximate_words)

    am_linking_dict = _time_regex_match(r'\d+am',
                                        utterance,
                                        char_offset_to_token_index,
                                        lambda match: [int(match.rstrip('am'))
                                                       * HOUR_TO_TWENTY_FOUR],
                                        indices_of_approximate_words)

    oclock_linking_dict = _time_regex_match(r"\d+\so'clock",
                                            utterance,
                                            char_offset_to_token_index,
                                            lambda match: digit_to_query_time(match.rstrip("o'clock")),
                                            indices_of_approximate_words)

    times_linking_dict: Dict[str, List[int]] = defaultdict(list)
    linking_dicts = [pm_linking_dict, am_linking_dict, oclock_linking_dict]

    for linking_dict in linking_dicts:
        for key, value in linking_dict.items():
            times_linking_dict[key].extend(value)

    return times_linking_dict

def get_date_from_utterance(tokenized_utterance: List[Token],
                            year: int = 1993,
                            month: int = None,
                            day: int = None) -> datetime:
    """
    When the year is not explicitly mentioned in the utterance, the query assumes that
    it is 1993 so we do the same here. If there is no mention of the month or day then
    we do not return any dates from the utterance.
    """
    utterance = ' '.join([token.text for token in tokenized_utterance])
    year_result = re.findall(r'199[0-4]', utterance)
    if year_result:
        year = int(year_result[0])

    for token in tokenized_utterance:
        if token.text in MONTH_NUMBERS:
            month = MONTH_NUMBERS[token.text]
        if token.text in DAY_NUMBERS:
            day = DAY_NUMBERS[token.text]

    for tens, digits in zip(tokenized_utterance, tokenized_utterance[1:]):
        bigram = ' '.join([tens.text, digits.text])
        if bigram in DAY_NUMBERS:
            day = DAY_NUMBERS[bigram]
    if month and day:
        return datetime(year, month, day)
    return None

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

    number_linking_dict: Dict[str, List[int]] = defaultdict(list)
    for token_index, token in enumerate(tokenized_utterance):
        if token.text.isdigit():
            if token_index - 1 in indices_of_words_preceding_time:
                for time in digit_to_query_time(token.text):
                    number_linking_dict[str(time)].append(token_index)
            else:
                number_linking_dict[token.text].append(token_index)

    times_linking_dict = get_times_from_utterance(utterance,
                                                  char_offset_to_token_index,
                                                  indices_of_approximate_words)

    for key, value in times_linking_dict.items():
        number_linking_dict[key].extend(value)

    for index, token in enumerate(tokenized_utterance):
        for number in NUMBER_TRIGGER_DICT.get(token.text, []):
            number_linking_dict[number].append(index)

    for tens, digits in zip(tokenized_utterance, tokenized_utterance[1:]):
        bigram = ' '.join([tens.text, digits.text])
        if bigram in DAY_NUMBERS:
            number_linking_dict[str(DAY_NUMBERS[bigram])].append(len(tokenized_utterance) - 1)

    return number_linking_dict

def digit_to_query_time(digit: str) -> List[int]:
    """
    Given a digit in the utterance, return a list of the times that it corresponds to.
    """
    if int(digit) % 12 == 0:
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
        approximate_times.append((time + AROUND_RANGE) % HOURS_IN_DAY)
        # The number system is not base 10 here, there are 60 minutes
        # in an hour, so we can't simply add time - AROUND_RANGE.
        approximate_times.append((time - HOUR_TO_TWENTY_FOUR + AROUND_RANGE) % HOURS_IN_DAY)
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
                linking_scores_dict[str(query_value)].append(char_offset_to_token_index[match.start()])
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


MISC_TIME_TRIGGERS = {'morning': ['0', '1200'],
                      'afternoon': ['1200', '1800'],
                      'early afternoon' : ['1200', '1400'],
                      'after': ['1200', '1800'],
                      'evening': ['1800', '2200'],
                      'late evening': ['2000', '2200'],
                      'lunch': ['1400'],
                      'noon': ['1200']}

ALL_TABLES = {'aircraft': ['aircraft_code', 'aircraft_description',
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
                       'city' : ['city_name', 'state_code'],
                       'fare' : ['round_trip_required'],
                       'flight' : ['airline_code'],
                       'airport' : ['airport_code'],
                       'state' : ['state_name'],
                       'fare_basis' : ['fare_basis_code', 'class_type'],
                       'class_of_service' : ['booking_class'],
                       'aircraft' : ['basic_type', 'manufacturer'],
                       'restriction' : ['restriction_code'],
                       'ground_service' : ['transport_type'],
                       'days' : ['day_name']}

DAY_OF_WEEK = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']

DAY_OF_WEEK_INDEX = {idx : [day] for idx, day in enumerate(DAY_OF_WEEK)}

NUMBER_TRIGGER_DICT: Dict[str, List[str]] = get_trigger_dict([], [convert_to_string_list_value_dict(MONTH_NUMBERS),
                                                                  convert_to_string_list_value_dict(DAY_NUMBERS),
                                                                  MISC_TIME_TRIGGERS])
