
"""
  This code features the article

 "Pareto-based evaluation of national responses to COVID-19 pandemic shows
  that saving lives and protecting economy are non-trade-off objectives"

  by Kochanczyk & Lipniacki (Scientific Reports, 2021).

  This file contains shared auxiliary data and common settings.

  License: MIT
  Last changes: November 09, 2020
"""

import pandas as pd


FINAL_DAY = pd.to_datetime('2020-11-01')


MIN_POPULATION_M = 5


LOCATIONS = {
    'Europe': [
        'Italy', 'Spain','France', 'Belgium', 'Switzerland', 'United Kingdom', 'Sweden', 'Germany',
        'Czechia', 'Austria', 'Croatia', 'Serbia', 'Poland', 'Slovakia', 'Denmark', 'Norway',
        'Finland', 'Netherlands', 'Ireland', 'Slovenia', 'Portugal', 'Romania', 'Bulgaria',
        'Hungary', 'Greece'],
    'Asia': [
        'South Korea', 'Taiwan', 'Japan'],
    'Americas': [
        'Mexico', 'Brazil', 'Peru', 'Colombia', 'Chile', 'Argentina', 'Bolivia', 'Ecuador',
        'Canada'],
    'USA': [
        'California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia',
        'North Carolina', 'Michigan'],
    'Australia_and_New_Zealand': [
        'Australia', 'New Zealand']}


def is_USA_state(location):
    return location in LOCATIONS['USA']


STATE_TO_ABBREV = {
    'California':     'CA',  'Texas':    'TX',
    'Florida':        'FL',  'New York': 'NY',
    'Pennsylvania':   'PA',  'Illinois': 'IL',
    'Ohio':           'OH',  'Georgia':  'GA',
    'North Carolina': 'NC',  'Michigan': 'MI'
}

ABBREV_TO_STATE = {abbrev: state for state, abbrev in STATE_TO_ABBREV.items()}



EUROPEAN_COUNTRY_CODES = {
    'AL': 'Albania',
    'AT': 'Austria',
    'BA': 'Bosnia and Herzegovina',
    'BE': 'Belgium',
    'BG': 'Bulgaria',
    'CH': 'Switzerland',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DE': 'Germany',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'EL': 'Greece',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'IE': 'Ireland',
    'IS': 'Iceland',
    'IT': 'Italy',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'LV': 'Latvia',
    'ME': 'Montenegro',
    'MK': 'North Macedonia',
    'MT': 'Malta',
    'NL': 'Netherlands',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'RS': 'Serbia',
    'SE': 'Sweden',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
    'TR': 'Turkey',
    'UK': 'United Kingdom',
    'XK': 'Kosovo'
}


# Data sources:
#
# * most countries -- OECD
#   [https://data.oecd.org/pop/population.htm]
#
# * Ukraine, Serbia, Uruguay, Peru, Bolivia, Ecuador, Paraguay, Guatemala, Taiwan -- the World Bank
#   [via Google Search]
#
# * states of USA -- US Census Bureau
#   [https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html]
#
POPULATION_ = {
    'Austria':         8.838,
    'Belgium':        11.404,
    'Bulgaria':        7.025,
    'Croatia':         4.081,
    'Cyprus':          0.869,
    'Czechia':        10.626,
    'Denmark':         5.790,
    'Estonia':         1.322,
    'Finland':         5.516,
    'France':         66.942,
    'Germany':        82.914,
    'Greece':         10.726,
    'Hungary':         9.768,
    'Ireland':         4.857,
    'Italy':          60.422,
    'Latvia':          1.927,
    'Lithuania':       2.802,
    'Luxembourg':      0.608,
    'Malta':           0.484,
    'Netherlands':    17.232,
    'Norway':          5.312,
    'Poland':         38.413,
    'Portugal':       10.284,
    'Romania':        19.471,
    'Serbia':          6.98 ,
    'Slovakia':        5.447,
    'Slovenia':        2.070,
    'Spain':          46.733,
    'Sweden':         10.175,
    'Switzerland':     8.513,
    'Ukraine':        41.98 ,
    'United Kingdom': 66.436,

    'Russia':        144.491,
    'Japan':         126.443,
    'Taiwan':         23.78 ,
    'South Korea':    51.635,
    'Australia':      24.993,
    'New Zealand':     4.886,

    'Argentina':      44.495,
    'Bolivia':        11.35 ,
    'Brazil':        208.495,
    'Chile':          18.751,
    'Colombia':       49.834,
    'Ecuador':        17.08 ,
    'Guatemala':      17.25 ,
    'Mexico':        125.328,
    'Paraguay':        6.96 ,
    'Peru':           32.00 ,
    'Uruguay':         3.45 ,

    'Canada':         37.059,

    'California':     39.512,
    'Texas':          28.996,
    'Florida':        21.478,
    'New York':       19.454,
    'Pennsylvania':   12.802,
    'Illinois':       12.672,
    'Ohio':           11.689,
    'Georgia':        10.617,
    'North Carolina': 10.488,
    'Michigan':        9.987,
}


def population(location):
    assert location in POPULATION_
    return POPULATION_[location]


# Data source:
#
# * Office Holidays webpage [https://www.officeholidays.com],
#   note: collected for time span: January 02, 2020 -- October 31, 2020
#
USA_HOLIDAYS_ = ['2020-01-20', '2020-05-25', '2020-07-03', '2020-07-04', '2020-09-07', '2020-10-12']
HOLIDAYS_ = {
    'Italy':          ['2020-01-06', '2020-04-13', '2020-04-25', '2020-05-01', '2020-06-02',
                       '2020-08-15'],
    'Spain':          ['2020-01-06', '2020-04-10', '2020-05-01', '2020-06-24', '2020-08-15',
                       '2020-10-12'],
    'France':         ['2020-04-13', '2020-05-01', '2020-05-08', '2020-05-21', '2020-05-22',
                       '2020-06-01', '2020-07-14', '2020-08-15'],
    'Belgium':        ['2020-04-13', '2020-05-01', '2020-05-21', '2020-05-22', '2020-06-01',
                       '2020-07-21', '2020-08-15'],
    'Switzerland':    ['2020-04-10', '2020-04-13', '2020-05-01', '2020-05-21', '2020-06-01',
                       '2020-08-01', '2020-08-15', '2020-09-20'],
    'United Kingdom': ['2020-04-10', '2020-04-13', '2020-05-08', '2020-05-25'],
    'Germany':        ['2020-04-10', '2020-04-13', '2020-05-01', '2020-05-21', '2020-05-22',
                       '2020-06-01', '2020-06-11', '2020-10-03'],
    'Sweden':         ['2020-01-06', '2020-04-10', '2020-04-12', '2020-04-13', '2020-05-01',
                       '2020-05-21', '2020-05-22', '2020-05-31', '2020-06-06', '2020-06-19'],
    'Czechia':        ['2020-04-10', '2020-04-13', '2020-05-01', '2020-05-08', '2020-07-05',
                       '2020-07-06', '2020-09-28', '2020-10-28'],
    'Austria':        ['2020-01-06', '2020-04-13', '2020-05-01', '2020-05-21', '2020-06-01',
                       '2020-06-11', '2020-08-15', '2020-10-26'],
    'Croatia':        ['2020-01-06', '2020-04-13', '2020-05-01', '2020-06-11', '2020-06-22',
                       '2020-08-05', '2020-08-15'],
    'Serbia':         ['2020-01-02', '2020-01-07', '2020-02-15', '2020-02-17', '2020-04-17',
                       '2020-04-18', '2020-04-19', '2020-04-20', '2020-05-01', '2020-05-02',
                       '2020-05-09'],
    'Poland':         ['2020-01-06', '2020-04-12', '2020-04-13', '2020-05-01', '2020-05-03',
                       '2020-05-31', '2020-06-11', '2020-08-15'],
    'Slovakia':       ['2020-01-06', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-08',
                       '2020-07-05', '2020-08-29', '2020-09-01', '2020-09-15'],
    'Denmark':        ['2020-04-09', '2020-04-10', '2020-04-13', '2020-06-05', '2020-05-08',
                       '2020-05-21', '2020-06-01', ],
    'Norway':         ['2020-04-09', '2020-04-10', '2020-04-12', '2020-04-13', '2020-05-01',
                       '2020-05-17', '2020-05-21', '2020-05-31', '2020-06-01'],
    'Finland':        ['2020-01-06', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-21',
                       '2020-06-19', '2020-06-20'],
    'Netherlands':    ['2020-04-12', '2020-04-13', '2020-04-27', '2020-05-05', '2020-05-21',
                       '2020-05-31', '2020-06-01'],
    'Ireland':        ['2020-03-17', '2020-04-13', '2020-05-04', '2020-06-01', '2020-08-03',
                       '2020-10-26'],
    'Slovenia':       ['2020-01-02', '2020-02-08', '2020-04-12', '2020-04-13', '2020-04-27',
                       '2020-05-01', '2020-05-02', '2020-05-31', '2020-06-25', '2020-08-15',
                       '2020-10-31'],
    'Portugal':       ['2020-04-10', '2020-04-12', '2020-04-25', '2020-05-01', '2020-06-10',
                       '2020-06-11', '2020-08-15', '2020-10-05'],
    'Romania':        ['2020-01-02', '2020-01-24', '2020-04-17', '2020-04-20', '2020-05-01',
                       '2020-06-01', '2020-06-07', '2020-06-08', '2020-08-15'],
    'Bulgaria':       ['2020-03-03', '2020-04-17', '2020-04-18', '2020-04-19', '2020-04-20',
                       '2020-05-01', '2020-05-06', '2020-05-24', '2020-05-25', '2020-09-06',
                       '2020-09-07', '2020-09-22'],
    'Hungary':        ['2020-03-15', '2020-04-10', '2020-04-12', '2020-04-13', '2020-04-10',
                       '2020-05-01', '2020-05-31', '2020-06-01', '2020-08-20', '2020-08-21',
                       '2020-10-23'],
    'South Korea':    ['2020-01-24', '2020-01-25', '2020-01-26', '2020-03-01', '2020-04-15',
                       '2020-04-30', '2020-05-01', '2020-05-05', '2020-06-06', '2020-08-15',
                       '2020-08-17', '2020-09-30', '2020-10-01', '2020-10-02', '2020-10-03',
                       '2020-10-09'],
    'Greece':         ['2020-01-06', '2020-03-02', '2020-03-25', '2020-04-17', '2020-04-20',
                       '2020-05-01', '2020-06-08', '2020-08-15', '2020-10-28'],
    'Japan':          ['2020-01-13', '2020-02-11', '2020-02-23', '2020-02-24', '2020-03-20',
                       '2020-04-29', '2020-05-03', '2020-05-04', '2020-05-05', '2020-05-06',
                       '2020-07-23', '2020-07-24', '2020-08-10', '2020-09-21', '2020-09-22'],
    'Australia':      ['2020-01-27', '2020-03-09', '2020-04-10', '2020-04-13', '2020-04-25',
                       '2020-10-05'],


    'New Zealand':    ['2020-01-02', '2020-02-06', '2020-04-10', '2020-04-13', '2020-04-27',
                       '2020-06-01', '2020-10-26'],
    'Taiwan':         ['2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27',
                       '2020-01-28', '2020-01-29', '2020-02-28', '2020-04-02', '2020-04-03',
                       '2020-04-05', '2020-05-01', '2020-06-25', '2020-06-26', '2020-10-01',
                       '2020-10-02', '2020-10-09', '2020-10-10'],
    'Mexico':         ['2020-02-03', '2020-03-16', '2020-04-09', '2020-04-10', '2020-05-01',
                       '2020-09-16', '2020-10-12'],
    'Canada':         ['2020-02-17', '2020-05-18', '2020-07-01', '2020-09-07'],
    'Guatemala':      ['2020-04-09', '2020-04-10', '2020-04-11', '2020-05-01', '2020-06-29',
                       '2020-09-15', '2020-10-20'],
    'Brazil':         ['2020-02-24', '2020-02-25', '2020-04-10', '2020-04-21', '2020-05-01',
                       '2020-06-11', '2020-09-07', '2020-10-12'],
    'Argentina':      ['2020-02-24', '2020-02-25', '2020-03-23', '2020-03-24', '2020-04-02',
                       '2020-04-10', '2020-05-01', '2020-05-25', '2020-06-15', '2020-06-20',
                       '2020-07-09', '2020-07-10', '2020-08-17', '2020-10-12'],
    'Chile':          ['2020-04-10', '2020-04-11', '2020-05-01', '2020-05-21', '2020-06-29',
                       '2020-07-16', '2020-08-15', '2020-09-18', '2020-09-19', '2020-10-12',
                       '2020-10-25'],
    'Peru':           ['2020-04-09', '2020-04-10', '2020-05-01', '2020-06-24', '2020-06-29',
                       '2020-07-27', '2020-07-28', '2020-07-29', '2020-08-30'],
    'Colombia':       ['2020-01-06', '2020-03-23', '2020-04-09', '2020-04-10', '2020-05-01',
                       '2020-05-25', '2020-06-15', '2020-07-20', '2020-08-07', '2020-08-17',
                       '2020-10-12'],
    'Bolivia':        ['2020-01-22', '2020-02-24', '2020-02-25', '2020-04-10', '2020-05-01',
                       '2020-06-11', '2020-06-21', '2020-06-22', '2020-08-06'],
    'Ecuador':        ['2020-02-24', '2020-02-25', '2020-04-10', '2020-04-12', '2020-05-01',
                       '2020-05-24', '2020-05-25', '2020-08-10', '2020-10-09'],
    'California':     USA_HOLIDAYS_,
    'Texas':          USA_HOLIDAYS_,
    'Florida':        USA_HOLIDAYS_,
    'New York':       USA_HOLIDAYS_,
    'Pennsylvania':   USA_HOLIDAYS_,
    'Illinois':       USA_HOLIDAYS_,
    'Ohio':           USA_HOLIDAYS_,
    'Georgia':        USA_HOLIDAYS_,
    'North Carolina': USA_HOLIDAYS_,
    'Michigan':       USA_HOLIDAYS_,
}

HOLIDAYS = {country: set(map(pd.to_datetime, days)) for country, days in HOLIDAYS_.items()}



# probability density for time since the S-->E transition to the I-->R transition
# (hypoexponential distribution for days [0..21])
INFECTION_TO_REMOVAL = [
    0.,
    0.000362979,
    0.00850114,
    0.0358891,
    0.0758174,
    0.11051,
    0.128365,
    0.128427,
    0.115997,
    0.0975333,
    0.0779709,
    0.0601564,
    0.0452763,
    0.0335013,
    0.0245049,
    0.0177883,
    0.012849,
    0.00925193,
    0.00664881,
    0.00477236,
    0.003423,
    0.00245411
]


# probability density for time since S-->E to death
# (distribution for days [0..42])
INFECTION_TO_DEATH = [
    0.,
    1.97992*10**-15,
    4.96784*10**-10,
    2.08575*10**-7,
    8.30308*10**-6,
    0.00009960299186504467,
    0.0005843509115979334,
    0.002144972702774499,
    0.005674948293474557,
    0.011821691182246607,
    0.020562910676718952,
    0.0311138639171267,
    0.04218413436293671,
    0.05239136621143307,
    0.060619542226827285,
    0.06620925747143831,
    0.06897657769841008,
    0.06911812710095659,
    0.06707087474999178,
    0.06337751970378437,
    0.05858375211356976,
    0.05317391580972261,
    0.0475401237406161,
    0.041975439613327496,
    0.036681677652649104,
    0.03178431842679064,
    0.027349453777706586,
    0.023399783883864624,
    0.019928234941009117,
    0.016908758018002484,
    0.01430442844254012,
    0.01207322739806098,
    0.010171968470267148,
    0.008558814593069312,
    0.007194768774661015,
    0.006044446509591807,
    0.0050763653251489245,
    0.004262924687770805,
    0.003580199535760508,
    0.003007632412087736,
    0.002527680841548896,
    0.002125456244188818,
    0.0017883764379078187
]


# common sliding window settings
ROLL_OPTS = {
    'closed': 'both',
    'win_type': 'boxcar',
    'center': True
}
