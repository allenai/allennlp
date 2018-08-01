from typing import List, Dict
from datetime import datetime
import re

from collections import defaultdict
from allennlp.data.tokenizers import Token

TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100
HOURS_IN_DAY = 2400
AROUND_RANGE = 30

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

def get_numbers_from_utterance(utterance: str) -> List[str]:
    """
    Given an utterance, find all the numbers that are in the action space.
    """
    numbers = []
    numbers.extend(re.findall(r'\d+', utterance))
    numbers.extend([str(int(num_str.rstrip('pm')) * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR)
                    for num_str in re.findall(r'\d+', utterance)])

    numbers.extend(get_times_from_utterance(utterance))

    words = utterance.split(' ')
    for word in words:
        if word in MONTH_NUMBERS:
            numbers.append(str(MONTH_NUMBERS[word]))
        if word in DAY_NUMBERS:
            numbers.append(str(DAY_NUMBERS[word]))
        if word in MISC_TIME_TRIGGERS:
            numbers.extend(MISC_TIME_TRIGGERS[word])

    for tens, digits in zip(words, words[1:]):
        day = ' '.join([tens, digits])
        if day in DAY_NUMBERS:
            numbers.append(str(DAY_NUMBERS[day]))
    return sorted(numbers, reverse=True)

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

GROUND_SERVICE = {'air taxi': ['AIR TAXI OPERATION'],
                  'car': ['RENTAL CAR'],
                  'limo': ['LIMOUSINE'],
                  'rapid': ['RAPID TRANSIT'],
                  'rental': ['RENTAL CAR'],
                  'taxi': ['TAXI']}

MISC_STR = {"every day" : ["DAILY"]}

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

DAY_OF_WEEK_DICT = {'weekdays' : ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY']}

YES_NO = {'one way': ['NO'],
          'economy': ['YES']}

CITY_AIRPORT_CODES = {'atlanta' : ['ATL'],
                      'boston' : ['BOS'],
                      'baltimore': ['BWI'],
                      'charlotte': ['CLT'],
                      'dallas': ['DFW'],
                      'detroit': ['DTW'],
                      'la guardia': ['LGA'],
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

DAY_OF_WEEK_INDEX = {idx : [day] for idx, day in enumerate(DAY_OF_WEEK)}

TRIGGER_LISTS = [CITIES, AIRPORT_CODES,
                 STATES, STATE_CODES,
                 FARE_BASIS_CODE, CLASS,
                 AIRLINE_CODE_LIST, DAY_OF_WEEK,
                 CITY_CODE_LIST, MEALS,
                 RESTRICT_CODES]

TRIGGER_DICTS = [CITY_AIRPORT_CODES,
                 AIRLINE_CODES,
                 CITY_CODES,
                 GROUND_SERVICE,
                 DAY_OF_WEEK_DICT,
                 YES_NO,
                 MISC_STR]

ATIS_TRIGGER_DICT = get_trigger_dict(TRIGGER_LISTS, TRIGGER_DICTS)
