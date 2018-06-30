from typing import List
import re

TWELVE_TO_TWENTY_FOUR = 1200
HOUR_TO_TWENTY_FOUR = 100

def get_times_from_utterance(utterance: str) -> List[str]:
    """
    Given an utterance, get the numbers that correspond to times and convert time 
    for example: convert ``7pm`` to 1900
    """
    pm_times = [pm_str.rstrip('pm') * HOUR_TO_TWENTY_FOUR + TWELVE_TO_TWENTY_FOUR for pm_str in re.findall(r'\d+pm', utterance)]
    am_times = [am_str.rstrip('am') * HOUR_TO_TWENTY_FOUR for am_str in re.findall(r"\d+am", utterance)]
    oclock_times = [oclock_str.rstrip("o'clock") * HOUR_TO_TWENTY_FOUR for oclock_str in re.findall(r"\d+\so'clock", utterance)]
    return am_times + pm_times + oclock_times


def get_nums_from_utterance(utterance: str) -> List[str]:
    """
    Given an utterance, find all the numbers that are in the action space.
    """
    nums = ['1', '0']
    nums.extend(re.findall(r'\d+', utterance))
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
        'twelth': 12,
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

MISC_TIME_TRIGGERS = {
        'morning': ['0', '1200'],
        'afternoon': ['1200', '1800'],
        'lunch': ['1400']
        }


        
CITIES = ['NASHVILLE', 'BOSTON', 'BURBANK', 'BALTIMORE', 'CHICAGO', 'CLEVELAND', 'CHARLOTTE', 'COLUMBUS', 'CINCINNATI', 'DENVER', 'DALLAS', 'DETROIT', 'FORTWORTH', 'HOUSTON', 'WESTCHESTER COUNTY', 'INDIANAPOLIS', 'NEWARK', 'LAS VEGAS', 'LOS ANGELES', 'LONG BEACH', 'ATLANTA', 'MEMPHIS', 'MIAMI', 'KANSAS CITY', 'MILWAUKEE', 'MINNEAPOLIS', 'NEW YORK', 'OAKLAND', 'ONTARIO', 'ORLANDO', 'PHILADELPHIA', 'PHOENIX', 'PITTSBURGH', 'ST. PAUL', 'SAN DIEGO', 'SEATTLE', 'SAN FRANCISCO', 'SAN JOSE', 'SALT LAKECITY', 'ST. LOUIS', 'ST. PETERSBURG', 'TACOMA', 'TAMPA', 'WASHINGTON', 'MONTREAL', 'TORONTO']

AIRLINE_CODES = {'argentina': 'AR',
                 'alliance': '3J',
                 'canada': 'AC',
                 'ontario': 'GX',
                 'wisconson': 'ZW',
                 'alaska': 'AS',
                 'alpha': '7V',
                 'american.': 'AA',
                 'american': 'TZ',
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
                 'grand': 'QD',
                 'lufthansa': 'LH',
                 'mesaba': 'XJ',
                 'mgm': 'MG',
                 'midwest': 'YX',
                 'nation': 'NX',
                 'northeast': '2V',
                 'nortwest': 'NW',
                 'ontario': '9X',
                 'precision': 'RP',
                 'royal': 'AT',
                 'sabena': 'SN',
                 'sky': 'OO',
                 'south': 'WN',
                 'thai': 'TG',
                 'tower': 'FF',
                 'trans': '9N',
                 'trans': 'TW',
                 'trans': 'RZ',
                 'united': 'UA',
                 'us': 'US',
                 'west': 'OE'}

DEFAULT_TABLES = ['airport_service', 'city', 'flight']

TABLES = {'aircraft': ['aircraft_code', 'aircraft_description', 'manufacturer', 'basic_type', 'propulsion', 'wide_body', 'pressurized'],
          'airline': ['airline_name', 'airline_code'],
          'airport': ['airport_code', 'airport_name', 'airport_location', 'state_code', 'country_name', 'time_zone_code', 'minimum_connect_time'],
          'airport_service': ['city_code', 'airport_code', 'miles_distant', 'direction', 'minutes_distant'],
          'city': ['city_code', 'city_name', 'state_code', 'country_name', 'time_zone_code'],
           'date_day': ['month_number', 'day_number', 'year', 'day_name'],
           'days': ['days_code', 'day_name'],
           'equipment_sequence': ['aircraft_code_sequence', 'aircraft_code'],
           'fare': ['faire_id', 'from_airport', 'to_airport', 'fare_basis_code', 'fare_airline', 'restriction_code', 'one_direction_cost', 'round_trip_cost', 'round_trip_required'],
           'fare_basis': ['fare_basis_code', 'booking_class', 'class_type', 'premium', 'economy', 'discounted', 'night', 'season', 'basis_days'],
           'flight': ['flight_id', 'flight_days', 'from_airport', 'to_airport', 'departure_time', 'arrival_time', 'airline_flight', 'airline_code', 'flight_number',
               'aircraft_code_sequence', 'meal_code', 'stops', 'connections', 'dual_carrier', 'time_elapsed'],
           'flight_fare': ['flight_id', 'flight_fare'],
           'flight_leg': ['flight_id', 'leg_number', 'leg_flight'],
           'food_service': ['meal_code', 'meal_number', 'compartment', 'meal_description'],
           'ground_service': ['city_code', 'airport_code', 'transport_type', 'ground_fare'],
           'month': ['month_number', 'month_name'],
           'restriction': ['restriction_code', 'advance_purchase', 'stopovers', 'saturday_stay_required', 'minimum_say', 'maximum_stay', 'application', 'no_discounts'],
           'state': ['state_code', 'state_name', 'country_name']}

