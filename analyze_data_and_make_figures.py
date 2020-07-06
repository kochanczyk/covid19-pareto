#!/usr/bin/env python3
"""
  This code features the article "Evaluation of national responses to COVID-19 pandemic based
  on Pareto optimality" by Kochanczyk & Lipniacki (submitted to Scientific Reports, 2020).

  Author: Marek Kochanczyk
  License: MIT
  Last changes: June 26, 2020
"""

import re
from operator import itemgetter
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
import matplotlib.patheffects as pthff
from colorsys import rgb_to_hls
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# -- Plot shared settings --------------------------------------------------------------------------

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.pad'] = 1.67
plt.rcParams['ytick.major.pad'] = 1.33
plt.rc('font', size=8, family='sans-serif')
plt.rc('text', usetex=True)
plt.rc('text.latex',
       preamble=r'''\usepackage[svgnames]{xcolor}\usepackage[T1]{fontenc}\usepackage{cmbright}''')


# -- Data analysis shared settings -----------------------------------------------------------------

ROLL_OPTS = {
    'closed': 'both',
    'win_type': 'boxcar',
    'center': True
}


# distribution for days [0 : 21]
HYPOEXP = [
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

# distribution for days [0..42]
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

POPULATION = {
    'Portugal':       10.28,
    'Spain':          46.94,
    'France':         66.99,
    'United Kingdom': 66.65,
    'Belgium':        11.46,
    'Netherlands':    17.28,
    'Italy':          60.36,
    'Germany':        83.02,
    'Poland':         37.97,
    'Austria':         8.86,
    'Switzerland':     8.57,
    'Hungary':         9.77,
    'Slovenia':        2.08,
    'Slovakia':        5.45,
    'Czechia':        10.65,
    'Denmark':         5.81,
    'Ireland':         4.91,
    'Finland':         5.52,
    'Sweden':         10.23,
    'Norway':          5.37,
    'Ukraine':        41.98,
    'Romania':        19.41,
    'Bulgaria':        7.00,
    'Croatia':         4.08,
    'Greece':         10.72,
    'Serbia':          6.98,
    'South Korea':    51.64,
    'Russia':        144.50,
    'Canada':         37.59,
    'Brazil':        209.50,
    'Argentina':      44.50,
    'Chile':          18.73,
    'Uruguay':         3.45,
    'Peru':           32.00,
    'Colombia':       49.65,
    'Bolivia':        11.35,
    'Ecuador':        17.08,
    'Paraguay':        6.96,
    'Guatemala':      17.25,
    'Mexico':        126.20,
    'Japan':          126.5,
    'Australia':      24.99,
    'New Zealand':     4.89,
    'Taiwan':         23.78,
    'California':     39.51,
    'Texas':          29.00,
    'Florida':        21.48,
    'New York':       19.45,
    'Pennsylvania':   12.80,
    'Illinois':       12.67,
    'Ohio':           11.69,
    'Georgia':        10.62,
    'North Carolina': 10.49,
    'Michigan':        9.99
}

STATE_TO_ABBREV = {
    'California':    'CA',
    'Texas':         'TX',
    'Florida':       'FL',
    'New York':      'NY',
    'Pennsylvania':  'PA',
    'Illinois':      'IL',
    'Ohio':          'OH',
    'Georgia':       'GA',
    'North Carolina':'NC',
    'Michigan':      'MI'
}

ABBREV_TO_STATE = {abbrev: state for state, abbrev in STATE_TO_ABBREV.items()}



USE_DATA_SNAPSHOT = True

if USE_DATA_SNAPSHOT:
    SNAPSHOT_BASE_URL = 'https://raw.githubusercontent.com/' + \
                        'kochanczyk/covid19-pareto/master/data/snapshot-20200706/'

    OWID_DATA_URL = SNAPSHOT_BASE_URL + 'owid-covid-data.csv.bz2'
    OWID_TESTING_DATA_URL = SNAPSHOT_BASE_URL + 'covid-testing-all-observations.csv.bz2'
    MOBILITY_DATA_URL = SNAPSHOT_BASE_URL + 'Global_Mobility_Report.csv.bz2'
    TRACKING_URL = SNAPSHOT_BASE_URL + 'daily.csv.bz2'
else:
    OWID_DATA_URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    OWID_TESTING_DATA_URL = 'https://raw.githubusercontent.com/owid/covid-19-data/' + \
                            'master/public/data/testing/covid-testing-all-observations.csv'
    MOBILITY_DATA_URL = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    TRACKING_URL = 'https://covidtracking.com/api/v1/states/daily.csv'



# from Jan 02 till June 30, source: https://www.officeholidays.com
USA_HOLIDAYS_ = ['2020-01-20', '2020-05-25']
HOLIDAYS_ = {
    'Italy':          ['2020-01-06', '2020-04-13', '2020-04-25', '2020-05-01', '2020-06-02'],
    'Spain':          ['2020-01-06', '2020-04-10', '2020-05-01'],
    'France':         ['2020-04-13', '2020-05-01', '2020-05-08', '2020-05-21', '2020-06-01'],
    'Belgium':        ['2020-04-13', '2020-05-01', '2020-05-21', '2020-06-01'],
    'Switzerland':    ['2020-04-10', '2020-04-13', '2020-05-21', '2020-06-01'],
    'United Kingdom': ['2020-04-10', '2020-05-08', '2020-05-25'],
    'Germany':        ['2020-04-10', '2020-04-13', '2020-05-01', '2020-05-21', '2020-06-01'],
    'Sweden':         ['2020-01-06', '2020-04-10', '2020-04-12', '2020-04-13', '2020-05-01',
                       '2020-05-21', '2020-05-31', '2020-06-06'],
    'Czechia':        ['2020-04-10', '2020-04-13', '2020-05-01', '2020-05-08'],
    'Austria':        ['2020-01-06', '2020-04-13', '2020-05-01', '2020-05-21', '2020-06-01',
                       '2020-06-11'],
    'Croatia':        ['2020-01-06', '2020-04-13', '2020-05-01', '2020-06-11', '2020-06-22'],
    'Serbia':         ['2020-01-02', '2020-01-07', '2020-02-15', '2020-02-17', '2020-04-17',
                       '2020-04-18', '2020-04-19', '2020-04-20', '2020-05-01', '2020-05-02',
                       '2020-05-09'],
    'Poland':         ['2020-01-06', '2020-04-12', '2020-04-13', '2020-05-01', '2020-05-03',
                       '2020-05-31', '2020-06-11'],
    'Slovakia':       ['2020-01-06', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-08'],
    'Denmark':        ['2020-04-09', '2020-04-10', '2020-04-13', '2020-05-08', '2020-05-21',
                       '2020-06-01'],
    'Norway':         ['2020-04-09', '2020-04-10', '2020-04-12', '2020-04-13', '2020-05-01',
                       '2020-05-17', '2020-05-21', '2020-05-31', '2020-06-01'],
    'Finland':        ['2020-01-06', '2020-04-10', '2020-04-13', '2020-05-01', '2020-05-21',
                       '2020-06-19', '2020-06-20'],
    'Netherlands':    ['2020-04-12', '2020-04-13', '2020-04-27', '2020-05-05', '2020-05-21',
                       '2020-05-31', '2020-06-01'],
    'Ireland':        ['2020-03-17', '2020-04-13', '2020-05-04', '2020-06-01'],
    'Slovenia':       ['2020-01-02', '2020-02-08', '2020-04-12', '2020-04-13', '2020-04-27',
                       '2020-05-01', '2020-05-02', '2020-05-31', '2020-06-25'],
    'Portugal':       ['2020-04-10', '2020-04-12', '2020-04-25', '2020-05-01', '2020-06-10',
                       '2020-06-11'],
    'Romania':        ['2020-01-02', '2020-01-24', '2020-04-17', '2020-04-20', '2020-05-01',
                       '2020-06-01', '2020-06-07', '2020-06-08'],
    'Bulgaria':       ['2020-03-03', '2020-04-17', '2020-04-18', '2020-04-19', '2020-04-20',
                       '2020-05-01', '2020-05-06', '2020-05-24', '2020-05-25'],
    'Hungary':        ['2020-03-15', '2020-04-10', '2020-04-12', '2020-04-13', '2020-04-10',
                       '2020-05-01', '2020-05-31', '2020-06-01'],
    'South Korea':    ['2020-01-24', '2020-01-25', '2020-01-26', '2020-03-01', '2020-04-15',
                       '2020-04-30', '2020-05-01', '2020-05-05', '2020-06-06'],
    'Greece':         ['2020-01-06', '2020-03-02', '2020-03-25', '2020-04-17', '2020-04-20',
                       '2020-05-01', '2020-06-08'],
    'Japan':          ['2020-01-13', '2020-02-11', '2020-02-23', '2020-02-24', '2020-03-20',
                       '2020-04-29', '2020-05-03', '2020-05-04', '2020-05-05', '2020-05-06'],
    'Australia':      ['2020-01-27', '2020-04-10', '2020-04-13', '2020-04-25'],
    'New Zealand':    ['2020-01-02', '2020-02-06', '2020-04-10', '2020-04-13', '2020-04-27',
                       '2020-06-01'],
    'Taiwan':         ['2020-01-23', '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27',
                       '2020-01-28', '2020-01-29', '2020-02-28', '2020-04-02', '2020-04-03',
                       '2020-04-05', '2020-05-01', '2020-06-25', '2020-06-25'],
    'Mexico':         ['2020-02-03', '2020-03-16', '2020-04-09', '2020-04-10', '2020-05-01'],
    'Canada':         ['2020-02-17', '2020-05-18'],
    'Guatemala':      ['2020-04-09', '2020-04-10', '2020-04-11', '2020-05-01', '2020-06-29'],
    'Brazil':         ['2020-02-24', '2020-02-25', '2020-04-10', '2020-04-21', '2020-05-01',
                       '2020-06-11'],
    'Argentina':      ['2020-02-24', '2020-02-25', '2020-03-23', '2020-03-24', '2020-04-02',
                       '2020-04-10', '2020-05-01', '2020-05-25', '2020-06-15', '2020-06-20'],
    'Chile':          ['2020-04-10', '2020-04-11', '2020-05-01', '2020-05-21', '2020-06-29'],
    'Peru':           ['2020-04-09', '2020-04-10', '2020-05-01', '2020-06-24', '2020-06-29'],
    'Colombia':       ['2020-01-06', '2020-03-23', '2020-04-09', '2020-04-10', '2020-05-01',
                       '2020-05-25', '2020-06-15'],
    'Bolivia':        ['2020-01-22', '2020-02-24', '2020-02-25', '2020-04-10', '2020-05-01',
                       '2020-06-11', '2020-06-21', '2020-06-22'],
    'Ecuador':        ['2020-02-24', '2020-02-25', '2020-04-10', '2020-04-12', '2020-05-01',
                       '2020-05-24', '2020-05-25'],
    'California':     USA_HOLIDAYS_,
    'Texas':          USA_HOLIDAYS_,
    'Florida':        USA_HOLIDAYS_,
    'New York':       USA_HOLIDAYS_,
    'Pennsylvania':   USA_HOLIDAYS_,
    'Illinois':       USA_HOLIDAYS_,
    'Ohio':           USA_HOLIDAYS_,
    'Georgia':        USA_HOLIDAYS_,
    'North Carolina': USA_HOLIDAYS_,
    'Michigan':       USA_HOLIDAYS_
}

HOLIDAYS = {country: set(map(pd.to_datetime, days)) for country, days in HOLIDAYS_.items()}


THROWINS_ = {
    'Spain':          ['2020-04-19', '2020-05-22', '2020-05-25'],
    'France':         ['2020-05-07', '2020-05-29', '2020-06-03'],
    'United Kingdom': ['2020-05-21'],
    'Ireland':        ['2020-05-15'],
    'Portugal':       ['2020-05-03']
}

THROWINS = {country: list(map(pd.to_datetime, days)) for country, days in THROWINS_.items()}


# -- Plotting auxiliary functoins ------------------------------------------------------------------

def color_of(country, dull_color=(0.15, 0.15, 0.15)):
    colors = {
        'United Kingdom': (0.20, 0.00, 0.99),
        'Austria':        plt.cm.tab10(6),
        'Italy':          plt.cm.tab10(2),
        'Denmark':        (0.95, 0.15, 0.05),
        'Czechia':        plt.cm.tab10(4),
        'Sweden':         (0.10, 0.20, 0.90),
        'Belgium':        plt.cm.tab10(5),
        'Poland':         (0.15, 0.65, 1.00),
        'France':         (0.95, 0.25, 0.75),
        'Spain':          plt.cm.tab10(3),
        'Germany':        (0.55, 0.25, 0.70),
        'Switzerland':    (0.80, 0.35, 0.95),
        'Slovakia':       (0.25, 0.90, 0.50),
        'Russia':         (0.80, 0.45, 0.15),
        'Greece':         (0.45, 0.75, 1.00),
        'Norway':         plt.cm.tab10(0),
        'Slovenia':       plt.cm.tab10(1),
        'Romania':        plt.cm.tab10(8),
        'Portugal':       (0.90, 0.65, 0.00),
        'Finland':        plt.cm.tab10(9),
        'Netherlands':    (0.75, 0.50, 0.00),
        'Ireland':        (0.10, 0.80, 0.00),
        'Hungary':        (0.35, 0.35, 0.35),
        'Croatia':        (0.50, 0.55, 0.00),
        'Serbia':         (0.70, 0.60, 0.65),
        'Bulgaria':       plt.cm.tab10(2),
        'California':     (0.90, 0.70, 0.00),
        'Texas':          (0.35, 0.40, 0.40),
        'Florida':        (0.95, 0.40, 0.00),
        'New York':       (0.25, 0.00, 1.00),
        'Pennsylvania':   (0.20, 0.25, 1.00),
        'Illinois':       (0.80, 0.50, 0.00),
        'Ohio':           (0.65, 0.00, 0.00),
        'Georgia':        (0.00, 0.45, 0.80),
        'North Carolina': (0.10, 0.00, 0.95),
        'Michigan':       (0.05, 0.50, 0.15),
        'Brazil':         (0.00, 0.70, 0.20),
        'Mexico':         (0.00, 0.50, 0.60),
        'Peru':           (0.75, 0.50, 0.25),
        'Ecuador':        (0.65, 0.65, 0.00),
        'Chile':          (0.65, 0.15, 0.00),
        'Bolivia':        (0.20, 0.65, 0.00),
        'Colombia':       (0.00, 0.10, 0.65),
        'Argentina':      (0.30, 0.75, 1.00),
        'Guatemala':      (0.80, 0.10, 0.60),
        'Canada':         (0.80, 0.10, 0.60)
    }
    if country in colors.keys():
        return colors[country]
    else:
        return dull_color



def adjust_spines(ax, spines, left_shift=15, bottom_shift=0):
    for loc, spine in ax.spines.items():
        if loc in spines:
            if loc == 'left':
                spine.set_position(('outward', left_shift))
            elif loc == 'bottom':
                spine.set_position(('outward', bottom_shift))
        else:
            spine.set_color('none')

    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])


def set_ticks_lengths(ax):
    ax.tick_params(which='major', length=2., labelsize=7)
    ax.tick_params(which='minor', length=1.)


def abbrev_date(date):
    return str(date).replace('2020', '').split()[0] \
           .replace('-02-', 'Feb ').replace('-03-', 'Mar ') \
           .replace('-04-', 'Apr ').replace('-05-', 'May ') \
           .replace('-06-', 'Jun ')


def darken(color, scale=0.5):
    lightness = min(1, rgb_to_hls(*color[0:3])[1] * scale)
    return sns.set_hls_values(color=color, h=None, l=lightness, s=None)


# -- Data analysis auxiliary functions -------------------------------------------------------------

def is_USA_state(location):
    return location in LOCATIONS['USA']


def extract_cases_and_deaths(location, columns=['date', 'new_cases', 'total_cases',
                                                'new_deaths', 'total_deaths']):
    if is_USA_state(location):
        state_abbrev = STATE_TO_ABBREV[location]
        return TRACKING_DATA.loc[state_abbrev]
    else:
        country = location
        country_indices = OWID_DATA['location'] == country
        return OWID_DATA[country_indices][columns].set_index('date')

def extract_mobility(location):
    if is_USA_state(location):
        df = MOBILITY_DATA[ (MOBILITY_DATA['location'] == 'United States') & \
                            (MOBILITY_DATA['sub_region_1'] == location) & \
                             MOBILITY_DATA['sub_region_2'].isnull() ].set_index('date')
    else:
        df = MOBILITY_DATA[  (MOBILITY_DATA['location'] == location) \
                           & MOBILITY_DATA['sub_region_1'].isnull() ].set_index('date')
        assert df['sub_region_1'].isnull().all() and df['sub_region_2'].isnull().all()
    return df

def smoothed_daily_data(location, daily_ws=[3, 7, 14], fix=True):
    df = extract_cases_and_deaths(location).copy()

    if fix:
        # general
        for col_new in ['new_cases', 'new_deaths']:
            df[col_new] = df[col_new].fillna(0)
        for col_tot in ['total_cases', 'total_deaths']:
            if pd.isna(df.iloc[0][col_tot]):
                initial = df.index[0]
                df.at[initial, col_tot] = 0
            df[col_tot] = df[col_tot].ffill()

        # location-specific
        if location in THROWINS:
            for throwin in THROWINS[location]:
                new_cases = df.loc[throwin, 'new_cases']
                if new_cases == 0:
                    pass
                elif new_cases < 0:
                    df.loc[throwin, 'new_cases'] = 0
                elif new_cases > 0:
                    prevv = df.loc[throwin - pd.offsets.Day(1), 'new_cases']
                    nextt = df.loc[throwin + pd.offsets.Day(1), 'new_cases']
                    df.loc[throwin, 'new_cases'] = int(round(0.5*(prevv + nextt)))
            # WARNING: because of the above, diff(cumulative total) != daily

    for k in ['cases', 'deaths']:
        for w in daily_ws:
            df[f"new_{k}{w}"] = df[f"new_{k}"].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()

    for col in ['new_cases', 'total_cases', 'new_deaths', 'total_deaths']:
        df[col] = df[col].astype('Int64')

    return df


def reasonable_span(df, kind, location_population,
                    trim_front=True, trim_back=False, trim_w=7, front_min_cumul=100):
    assert kind in ['cases', 'deaths']

    back_max_weekly_per_1M = {'cases': 20, 'deaths': 1}[kind]

    # -- front
    if trim_front:
        above_min_cumul_indices = df[f"total_cases"].values >= front_min_cumul
        df = df[above_min_cumul_indices]

    # -- back
    df_rev = df[f"new_{kind}"].reindex(index=df.index[::-1])
    df_rev = df_rev.rolling(window=trim_w, min_periods=trim_w//2+1, **ROLL_OPTS).sum()
    df_rev /= location_population
    if any(df_rev > back_max_weekly_per_1M) == True:
        green_day = df_rev[df_rev > back_max_weekly_per_1M].first_valid_index()
    else:
        green_day = df.index[0]

    if trim_back:
        if green_day is not None:
            df = df[:green_day]

    return (df, green_day)


def calc_Rt(theta):
    log2 = np.log(2)
    Td = log2/np.log(theta)
    m, n, sigma, gamma = 6, 1, 1/5.28, 1/3
    Rt = log2/Td*(log2/(Td * m *sigma) + 1)**m / (gamma*(1 - (log2/(Td * n * gamma) + 1)**(-n)))
    return Rt


def insert_epidemic_dynamics(df, timespan_days=14, data_smoothing_window=14):
    half_timespan_days = timespan_days//2
    for kind in ['cases', 'deaths']:
        values = df[f"new_{kind}{data_smoothing_window}"].values
        thetas = []
        for vi in range(len(values)):
            if vi < half_timespan_days or vi + half_timespan_days >= len(values):
                thetas += [pd.NA]
            else:
                expo = (1/(timespan_days - 1))
                theta = (values[vi + half_timespan_days] / values[vi - half_timespan_days])**expo
                theta = float(theta)
                if np.isnan(theta) or np.isinf(theta):
                    thetas += [pd.NA]
                else:
                    thetas += [theta]

        df[f"theta_{kind}"] = thetas
        df[f"Rt_{kind}"] = df[f"theta_{kind}"].map(calc_Rt)
    return df


def average_mobility_reduction(location_or_mo):
    if type(location_or_mo) == str:
        location = location_or_mo
        mo = extract_mobility(location)
    else:
        mo = location_or_mo
    return mo['retail_and_recreation workplaces'.split()].agg(np.mean, axis=1).astype('Float64').to_frame(name='mobility')


def insert_mobility_reduction(df, location, min_sum_weights=0.5):
    def has_day_(dd):         return dd in avg_mo.index
    def is_weekday_(dd):      return dd.dayofweek < 5
    def is_holiday_(cc, dd):   return cc in HOLIDAYS and dd in HOLIDAYS[cc]
    def is_valid_day_(cc, dd): return has_day_(dd) and is_weekday_(dd) and not is_holiday_(cc, dd)

    mo = extract_mobility(location)
    avg_mo = average_mobility_reduction(mo)
    df['mobility'] = avg_mo

    df['mobility_reduction'] = 0
    for day in mo.index:
        if day in df.index and is_valid_day_(location, day):
            df.at[day,'mobility_reduction'] = avg_mo.loc[day]

    for kind in ['cases', 'deaths']:
        distrib = {'cases': HYPOEXP, 'deaths': INFECTION_TO_DEATH}[kind]
        df[f"mobility_historical_{kind}"] = pd.NA  # previous values that gave rise to current daily new cases or deaths
        for day in mo.index:
            if day in df.index:
                valid_days_indices = {di for di in range(len(distrib))
                                      if is_valid_day_(location, day - pd.offsets.Day(di))}
                weights     = [distrib[di]
                               for di in valid_days_indices]
                weighted_ms = [distrib[di] * avg_mo.loc[day - pd.offsets.Day(di)]
                               for di in valid_days_indices]
                v = np.sum(weighted_ms) / np.sum(weights)
                df.at[day, f"mobility_historical_{kind}"] = \
                        v if np.sum(weights) >= min_sum_weights else pd.NA
    return df


def insert_tests_performed(df, location, interpolate=True, w=7, verbose=False):
    if is_USA_state(location):
        df[f"new_tests{w}"] = df['new_tests'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
        df['tests_per_hit'] = df[f"new_tests{w}"] \
                            / df['new_cases'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
        return df
    else:
        df_test = None

        entities = set(OWID_TESTING_DATA['Entity'])
        colnames = ['date', 'Cumulative total']
        for ending in ['tests performed', 'samples tested', 'units unclear', 'people tested',
                       'cases tested', 'samples analysed']:
            ent = f"{location.replace('Czechia', 'Czech Republic')} - {ending}"
            if ent in entities:
                if verbose: 
                    print(location, ent)
                ent_indices = OWID_TESTING_DATA['Entity'] == ent
                df_test = OWID_TESTING_DATA[ent_indices][colnames] \
                          .rename(columns={'Cumulative total': 'total_tests'}).set_index('date')
                break

        if df_test is None:
            print('Warning: no data on testing in', location)
            df['tests_per_hit'] = np.nan
            return df

        if interpolate:
            df_test['total_tests'] = df_test['total_tests'].interpolate(limit_area='inside',
                                                                        limit_direction='both')
        else:
            df_test['total_tests'] = df_test['total_tests'].astype('Int64')
        df_test['new_tests']     = df_test['total_tests'].diff()
        df_test[f"new_tests{w}"] = df_test['new_tests'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
        df_test['tests_per_hit'] = df_test[f"new_tests{w}"] \
                                 / df['new_cases'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()

        return df.join(df_test)


def process_location(location, kv=True):
    df = smoothed_daily_data(location)
    df = insert_epidemic_dynamics(df)
    df = insert_mobility_reduction(df, location)
    df = insert_tests_performed(df, location)
    return (location, df) if kv else df


# ==================================================================================================


OWID_DATA = pd.read_csv(OWID_DATA_URL, parse_dates=['date']) \
              .replace({'Czech Republic': 'Czechia'})
OWID_DATA['date'] = OWID_DATA['date'].apply(pd.to_datetime)

MOBILITY_DATA = pd.read_csv(MOBILITY_DATA_URL, parse_dates=['date'], low_memory=False)       \
                  .rename(columns=lambda colnm: re.sub('_percent_change_from_baseline$', '', colnm)) \
                  .rename(columns={'country_region': 'location'})

OWID_TESTING_DATA = pd.read_csv(OWID_TESTING_DATA_URL, parse_dates=['Date']) \
                      .rename(columns={'Date': 'date'})

TRACKING_DATA = pd.read_csv(TRACKING_URL, parse_dates=['date'])[::-1].set_index(['state', 'date'])
TRACKING_DATA['new_tests'] = TRACKING_DATA['negativeIncrease'] + TRACKING_DATA['positiveIncrease']
TRACKING_DATA['total_tests'] = TRACKING_DATA['negative'] + TRACKING_DATA['positive']
TRACKING_DATA.rename(columns={'positive': 'total_cases', 'positiveIncrease': 'new_cases',
                              'death': 'total_deaths',  'deathIncrease': 'new_deaths'}, inplace=True)


FINAL_DAY = pd.to_datetime('2020-06-17')
SPAIN_DAILY_FINAL_DAY = pd.to_datetime('2020-05-11')
MIN_POPULATION_M = 4

LOCATIONS = {
    'Europe': [
        'Italy', 'Spain','France', 'Belgium', 'Switzerland', 'United Kingdom', 'Sweden',  'Germany',
        'Czechia', 'Austria', 'Croatia', 'Serbia', 'Poland', 'Slovakia', 'Denmark', 'Norway',
        'Finland', 'Netherlands', 'Ireland', 'Slovenia','Portugal', 'Romania', 'Bulgaria', 'Hungary',
        'Greece'],
    'Asia': [
        'South Korea', 'Taiwan', 'Japan'],
    'Americas': [
        'Mexico', 'Brazil', 'Peru', 'Colombia', 'Chile', 'Argentina', 'Bolivia', 'Ecuador', 'Canada'],
    'USA': [
        'California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia',
        'North Carolina', 'Michigan'],
    'Australia': [
        'Australia', 'New Zealand']}

LOCATIONS_FLAT = [c for cs in LOCATIONS.values() for c in cs]

for c in LOCATIONS_FLAT:
    if c in LOCATIONS['USA']:
        assert STATE_TO_ABBREV[c] in TRACKING_DATA.index.get_level_values(0).unique()
    else:
        assert len(OWID_DATA[ OWID_DATA['location'] == c ])
        assert c in set(MOBILITY_DATA['location'])


TRAJS = dict(Pool(8).map(process_location, LOCATIONS_FLAT))

am_countries = [c for part in ['Americas', 'USA']
                  for c in LOCATIONS[part] if POPULATION[c] >= MIN_POPULATION_M]
eaa_countries = [c for part in ['Europe', 'Asia', 'Australia']
                   for c in LOCATIONS[part] if POPULATION[c] >= MIN_POPULATION_M]


# -- Figures 1 & 4 ---------------------------------------------------------------------------------

def pareto_front(data, optima=True):
    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=not optima)  # x-ascending
    front = [ sorted_data[0][2] ]
    cutoff = sorted_data[0][1]
    for sd in sorted_data[1:]:
        if (optima and sd[1] < cutoff) or (not optima and sd[1] > cutoff):
            front += [sd[2]]
            cutoff = sd[1]
    return front

def deaths_vs_mobility(trajs, country):
    mob = -trajs[country][['mobility_reduction']].resample('W').sum().cumsum()
    dth =  trajs[country][['new_deaths']].astype('Float64')
    dth = dth.resample('W').sum().cumsum() / POPULATION[country]
    df = mob.join(dth).rename(columns={'mobility_reduction':   f"mobility_cumul_{country}",
                                       'new_deaths': f"new_deaths_cumul_per_1M_{country}"})
    return df

def put_final_dot(ax, location, x, y, is_extra_country=False):
    label_shifts = { 'New Zealand':     (  5, 0.78),
                     'Ireland':         (  0, 0.8 ),
                     'France':          (  5, 0.86),
                     'Italy':           ( 25, 0.78),
                     'Spain':           ( 26, 1.06),
                     'United Kingdom':  ( 45, 1.07),
                     'Romania':         ( 20, 0.77),
                     'Finland':         ( 20, 0.77),
                     'Bulgaria':        (230, 0.71),
                     'Czechia':         (466, 0.88),
                     'Poland':          ( 5,  0.97),
                     'Denmark':         (130, 1.1 ),
                     'Norway':          (  0, 0.89),
                     'South Korea':     (  0, 0.81),
                     'Hungary':         (360, 1.08),
                     'Croatia':         (  0, 0.93),
                     'Mexico':          ( 26, 1.06),
                     'Canada':          ( 50, 1.11),
                     'Guatemala':       ( 15, 1.02),
                     'Argentina':       ( 20, 0.76),
                     'Florida':         ( 10, 0.9 ),
                     'Texas':           ( 10, 0.78),
                     'California':      ( 30, 0.72),
                     'Ohio':            (125, 0.68),
                     'Georgia':         ( 43, 1.07),
                     'Ecuador':         ( 50, 1.11),
                     'North Carolina':  (  0, 0.84),
                     'Illinois':        (340, 1.06),
                     'Pennsylvania':    ( 20, 0.75)}
    ax.plot([x[-1]], [y[-1]],  '-.', marker='8' if is_extra_country else 'o',
            linewidth=1, markersize=6.25, markeredgewidth=0, alpha=0.8, clip_on=False,
            color=color_of(location), markerfacecolor=color_of(location))
    loc = location if not is_USA_state(location) else r'\textit{\it ' + location + '}'
    loc = location.replace('United Kingdom', 'UK').replace('Spain', 'Spain*')
    ax.annotate(loc, xycoords='data',
                xy=(x[-1] + 52  - (0 if location not in label_shifts else label_shifts[location][0]),
                    y[-1]*1.033 * (1 if location not in label_shifts else label_shifts[location][1])),
                color=sns.set_hls_values(color_of(location), l=0.3), clip_on=False)



def jointly_trimmed_trajs(trajs, countries, cols, skipped=None, cleanup=True, force_end=FINAL_DAY,
                          verbose=False):

    assert len(cols) == 2
    col1, col2 = cols
    days_of_last_available_data = set()
    for cc in countries:
        if skipped and cc in skipped: continue
        df = trajs[cc]
        df_sel = df[ ~df[col1].isnull() & ~df[col2].isnull() ]
        last_day = df_sel.iloc[-1].name
        days_of_last_available_data.add(last_day)
        if verbose:
            print(cc, last_day.strftime('%b%d'))
    day_of_last_available_data = min(days_of_last_available_data)
    if force_end is None:
        if verbose:
            print(f"Last shared available day ({' & '.join(cols)}):",
                  day_of_last_available_data.strftime('%b%d'))
    else:
        if verbose:
            print(f"Last shared available day ({' & '.join(cols)}):",
                  day_of_last_available_data.strftime('%b%d'), '==FORCED=>',
                  force_end.strftime('%b%d'))
        day_of_last_available_data = force_end

    edited_trajs = {}
    assert len(cols) == 2
    for cc in trajs.keys():
        df = trajs[cc].loc[:day_of_last_available_data]
        edited_trajs[cc] = df[ ~df[col1].isnull() & ~df[col2].isnull() ] if cleanup else df

    return day_of_last_available_data, edited_trajs



def plot_flares(trajs, countries, fig_number='X', fronts=None):

    last_day, trajs = jointly_trimmed_trajs(trajs, countries, ['mobility', 'new_deaths'])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
    adjust_spines(ax, ['left', 'bottom'], left_shift=12)

    ax.set_xlim((0, 5e3))
    ax.set_ylim((0.1, 2e3))
    ax.set_xlabel(r'Cumulative lockdown \textcolor{Gray}' + str(r'{= $x$}' if fig_number=='1' else r'{}'))
    ax.set_yscale('log')
    ax.set_ylabel(r'Cumulative deaths /M \textcolor{Gray}' + str(r'{= $y$}' if fig_number=='1' else r'{}'))
    ax.set_yticklabels(['', 0.1, 1, 10, 100, 1000])   # MANUAL SHIFT ==> DOUBLE-CHECK FINAL FIGURE!

    extra_countries = ['Taiwan', 'South Korea', 'Japan', 'Australia', 'New Zealand']
    n_fronts = 4

    pareto_D, correlation_D = None, []

    for country in countries:
        if country in extra_countries:  continue
        d = deaths_vs_mobility(trajs, country)
        x, y = d.values.T
        put_final_dot(ax, country, x, y)
        ax.plot(x, y,  '-.', marker='o' , linestyle='-',
                linewidth=0.8, markersize=1.1, markeredgewidth=0, alpha=0.7,
                color=color_of(country), markerfacecolor=darken(color_of(country)))
        pareto_D = d if pareto_D is None else pareto_D.join(d)
        correlation_D += [ [x[-1], y[-1]] ]

    if fig_number == '1':
        for country in extra_countries:
            x, y = deaths_vs_mobility(trajs, country).values.T
            put_final_dot(ax, country, x, y, True)
            correlation_D += [ [x[-1], y[-1]] ]

    cs = [str(c).split('_')[-1] for c in pareto_D.columns][::2]
    pareto_D_rev = pareto_D.reindex(index=pareto_D.index[::-1])
    marker_style = {'markerfacecolor': None,
                    'fillstyle': 'none',
                    'markersize': 2.5,
                    'markeredgewidth': 0.35*0.9}

    # Pareto fronts
    if fronts is None:
        fronts = []
        for row_i, (_, row) in enumerate(pareto_D_rev.iterrows()):
            if row_i not in [2*i for i in range(n_fronts)]: continue
            d = row.values.reshape(row.values.shape[0]//2, 2)
            dd = [(*d[j], cs[j]) for j in range(len(d))]
            co = sns.set_hls_values('gray', l=0.2 + 0.075*row_i)
            for opt in [True, False]:
                front_countries = pareto_front(dd, opt)
                front = np.array([row[[f"mobility_cumul_{c}", f"new_deaths_cumul_per_1M_{c}"]].values
                                for c in front_countries])
                fronts += [front]

    for front_i, front in enumerate(fronts):
        co = sns.set_hls_values('gray', l=0.2 + 0.075*front_i)
        optimal = not (front_i % 2)
        ax.plot(*front.T, ':' if optimal else '--', c=co, linewidth=1.1, alpha=0.8)
        if fig_number == '1':
            for pt in front:
                ax.plot(*pt, marker='o' if optimal else 's', color=co, **marker_style)

    # Pearson correlation coefficient
    if fig_number == '1':
        rho = scipy.stats.pearsonr(       np.array(correlation_D)[:,0],
                                   np.log(np.array(correlation_D)[:,1]))[0]
        ax.annotate(r"Correlation: Pearson's $\rho(x, \log{y})$ = " + f"{rho:.2f}", xy=(0.60, 0.04),
                    xycoords='axes fraction', color='#666666')

    fig.tight_layout()
    fn = f"Figure{fig_number}_{last_day.strftime('%b%d')}.pdf"
    print('>>', fn)
    fig.savefig(fn)
    return (fig, fronts)



fig1, eaa_fronts = plot_flares(TRAJS, eaa_countries, fig_number='1')
fig5, fronts = plot_flares(TRAJS, am_countries, fig_number='5', fronts=eaa_fronts)



# -- Figures 2, 3, and S1 --------------------------------------------------------------------------

def put_legend_cases(ax_leg, color):

    z = [3, 10, 30, 100, 300, 1000, 3000, 10000]
    x = np.array(list(range(len(z))))
    y1 = np.ones(len(x))*0.62
    y2 = np.ones(len(x))*0.31
    y3 = np.ones(len(x))*0.0

    ax_leg.set_xlim((0 +0, len(z)-1 -0))
    ax_leg.set_ylim((0, 1))

    # tracer line
    for y in [y1, y2, y3]:
        xx = [float(x[0]) + 0.125] + list(x[1:-1]) + [float(x[-1]) - 0.125]
        ax_leg.plot(xx, y, linestyle='-', linewidth=0.75, alpha=1, solid_capstyle='round',
                    color='#ffaaee', clip_on=False, zorder=10)

    # variable thickness line (BEGIN)
    lwidths = [0.7 * (0 + np.log(z))]
    points = np.array([x, y1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    for segi, seg in enumerate(segments):
        seg = seg.T
        color = sns.set_hls_values(color, l=0.15 + (lwidths[0][segi] - 0.)/8)
        ax_leg.plot(seg[0]+0.05, seg[1], '-', color=color, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='butt', zorder=20, clip_on=False)

    # variable thickness line (END)
    points = np.array([x, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    for segi, seg in enumerate(segments):
        seg = seg.T
        el = min(1, 0.075 + ((lwidths[0][segi] - 0.)/7)**1.3)
        co = sns.set_hls_values('#77ffaa', l=el)
        ax_leg.plot(seg[0]+0.05, seg[1], '-', color=co, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='butt', zorder=20, clip_on=False)

    # dots + thin black
    for y in [y1, y2, y3]:
        xx, yy = x[:-1], y[:-1]
        ax_leg.scatter(xx + 0.5, yy, s=0.025, marker='o', facecolor='#000000', alpha=0.5,
                       clip_on=False, zorder=30)
        ax_leg.plot(xx + 0.5, yy, linestyle='--', linewidth=0.1, color='#000000', alpha=0.33,
                    clip_on=False, zorder=40)

    ax_leg.annotate(s=r'Tests per case:', xy=(0.5, 0.84), xycoords='axes fraction', fontsize=8,
                    ha="center", va="center")
    ax_leg.annotate(s=r'when \textbf{$>$ 20} new cases /week /M', xy=(0.5, 0.62-0.09),
                    xycoords='axes fraction', fontsize=6.5, ha="center", va="center")
    ax_leg.annotate(s=r'when \textbf{$<$ 20} new cases /week /M', xy=(0.5, 0.31-0.09),
                    xycoords='axes fraction', fontsize=6.5, ha="center", va="center")
    ax_leg.annotate(s=r'no data on testing', xy=(0.5, 0.055), xycoords='axes fraction',
                    fontsize=6.5, ha="center", va="center")

    for vi, v in enumerate(z):
        for y in [y1, y2]:
            extra_shift = -0.08 if v in [100, 300, 1000] else 0
            ax_leg.annotate(s=f"{v}"[::-1].replace('000', 'k')[::-1], color='black',
                            xy=(x[vi]+extra_shift + 0.5, y[vi]+0.05+0.005*vi), xycoords='data',
                            fontsize=5.75, ha="center", va="center", zorder=30, clip_on=False)


def put_legend_deaths(ax_leg, color):

    z = [1, 3, 10, 30, 100, 300]
    x = np.array(list(range(len(z))))

    y2 = np.ones(len(x))*0.37

    ax_leg.set_xlim((0-0.1, len(z)-1+0.1))
    ax_leg.set_ylim((0, 1))

    # variable thickness line (BEGIN)
    lwidths = [1*np.log(1 + np.array(z))]
    points = np.array([x, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    for segi, seg in enumerate(segments):
        seg = seg.T
        el = 0.1 + (lwidths[0][segi] - 0.)/14
        color = sns.set_hls_values(color, l=el)
        ax_leg.plot(seg[0]-0.025, seg[1], '-', color=color, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='butt',
                    zorder=20, clip_on=False)

    # dots + thin black
    for y in [y2]:
        xx, yy = x[:-1], y[:-1]
        ax_leg.scatter(xx + 0.5, yy, s=0.025, marker='o', facecolor='black', alpha=0.5,
                       clip_on=False, zorder=30)
        ax_leg.plot(xx + 0.5, yy, linestyle='--', linewidth=0.1, color='black', alpha=0.33,
                    clip_on=False, zorder=40)

    ax_leg.annotate(s=r'Cases per death:', xy=(0.5, 0.63), xycoords='axes fraction', fontsize=8,
                    ha="center", va="center")
    ax_leg.annotate(s=r'when \textbf{at least 1} new death /week /M', xy=(0.5, 0.22),
                    xycoords='axes fraction', fontsize=6.5, ha="center", va="center")

    for vi, v in enumerate(z):
        for y in [y2]:
            ax_leg.annotate(s=f"{v}", xy=(x[vi] + 0.5, y[vi]+0.05 + 0.005*vi), xycoords='data',
                            fontsize=6, ha="center", va="center", zorder=30, clip_on=False,
                            color='black')



def plot_pinworms(trajs, countries, kind, fig_number):
    assert kind in ['cases', 'deaths']

    skipped = ['Taiwan', 'Slovakia', 'New Zealand']
    mob_col, Rt_col = f"mobility_historical_{kind}", f"Rt_{kind}"
    last_day, trajs_edited = jointly_trimmed_trajs(trajs, countries, [mob_col, Rt_col],
                                                   skipped=skipped)

    def by_per_capita(cc):
        if kind == 'cases':
            assert last_day in trajs[cc].index, \
                    print(f"Day {last_day} not available for {cc} that ends on",
                          trajs[cc].tail(1).index)
            return trajs[cc].loc[last_day, f"total_{kind}"] / POPULATION[cc] + 1e6*is_USA_state(cc)
        else:
            if cc in skipped:
                return trajs[cc].iloc[-1][f"total_{kind}"] / 1e9 + 1e6*is_USA_state(cc)
            else:
                return trajs[cc].iloc[-1][f"total_{kind}"] / POPULATION[cc] + 1e6*is_USA_state(cc)
        return trajs[cc].loc[last_day, f"total_{kind}"] / POPULATION[cc] + 1e6*is_USA_state(cc)

    countries = sorted(countries, key=by_per_capita, reverse=True)

    trajs = trajs_edited

    if kind == 'cases':
        base_color = '#aabbdd'
        legend_fun = put_legend_cases
    else:
        base_color='#885500'
        legend_fun = put_legend_deaths
        days_back = 14

    facecolor = '#f8f6f4'
    ncols = 5
    nrows = (len(countries) + 1)//ncols
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8/5*ncols, 8/6*nrows))

    for ci, country in enumerate(countries):
        ax = fig.axes[ci]
        ax.set_facecolor(facecolor)

        if kind=='deaths' and country in skipped:
            ax.annotate(s=country, xy=(0.5, 0.88), xycoords='axes fraction', fontsize=9,
                        color='#666666', ha="center", va="center", clip_on=False, zorder=100)
            total = trajs[country].iloc[-1][f"total_{kind}"]
            ax.annotate(s="{:d} {:s} in total".format(int(round(total)), kind),
                        xy=(0.5, 0.77), xycoords='axes fraction', fontsize=6.5, color='#666666',
                        ha="center", va="center", clip_on=False, zorder=100)
            ax.annotate(s="(plot not shown)",
                        xy=(0.5, 0.67), xycoords='axes fraction', fontsize=6.5, color='#666666',
                        ha="center", va="center", clip_on=False, zorder=100)
            adjust_spines(ax, ['left', 'bottom'] if ax.is_first_col() else ['bottom'])
            ax.set_xticks(())
            continue

        row_i = ci//ncols
        if row_i == nrows-1:
            ax.set_xlabel('Mobility', labelpad=-1)
        ax.set_xlim((-100, 0))
        ax.set_xticks((-100, 0))
       #ax.xaxis.set_major_formatter(tckr.PercentFormatter(decimals=0))
        ax.set_xticklabels((r'$-100\%$', r'$0\%$'))

        if ax.is_first_col(): ax.set_ylabel(r'$R$')
        ax.set_ylim((0, 4))
        ax.yaxis.set_major_locator(tckr.MultipleLocator(1))
        ax.axhline(1, linestyle='--', linewidth=0.5, color='#666666')

        df, green_day = reasonable_span(trajs[country], kind, POPULATION[country]) ### <<<====
        if country == 'Spain':
            df = df.loc[:SPAIN_DAILY_FINAL_DAY]
            ax.annotate(s=f"(truncated on {SPAIN_DAILY_FINAL_DAY.strftime('%b %d')})",
                        xy=(0.5, 0.67), xycoords='axes fraction', fontsize=6.5, color='#666666',
                        ha="center", va="center", clip_on=False, zorder=100,
                        path_effects=[pthff.Stroke(linewidth=0.75, foreground=facecolor),
                                      pthff.Normal()])
        df_orig = df.copy()

        # tracer line
        if kind == 'cases':
            df = df_orig.copy()
            x, y = df[[mob_col, Rt_col]].values.T
            ax.plot(x, y, linestyle='-', linewidth=0.75, alpha=1,
                    solid_capstyle='round', color='#ffaaee', clip_on=True, zorder=10)

        # variable thickness line (BEGIN)
        df = df_orig.copy()
        if green_day is not None:
            df = df[:green_day + pd.offsets.Day(1)]

        x, y = df[[mob_col, Rt_col]].values.T
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if kind == 'cases':
            tests_per_hit = df['tests_per_hit'].values
            np.place(tests_per_hit, np.isinf(tests_per_hit) | (tests_per_hit > 10000), 10000)
            z = 0.7*np.log(0 + tests_per_hit)
            np.place(z, np.isnan(z), 0)
            np.place(z, np.isinf(z), 1000)
            np.place(z, z < 0,       0)
            lwidths = [z]
        else:
            de = df[['new_deaths14']]
            ca = df[['new_cases14' ]]
            ca = ca.set_index( ca.index.shift(+days_back, freq ='D') )  # <-- THIS
           #de = de.set_index( de.index.shift(-days_back, freq ='D') )  # <-- NEVER this
            z = de.join(ca)
            z['cases14_per_death14'] = z['new_cases14'] / z['new_deaths14']
            z = z['cases14_per_death14'].values
            np.place(z, np.isnan(z), 0)
            np.place(z, np.isinf(z), 1000)
            np.place(z, z < 0,       0)
            lwidths = [1*np.log(1 + z)]
        for segi, seg in enumerate(segments):
            seg = seg.T
            if kind == 'cases':  el = 0.15 + lwidths[0][segi] / 8
            else:                el = 0.1  + lwidths[0][segi] / 14
            co = sns.set_hls_values(base_color, l=el)
            ax.plot(seg[0], seg[1], '-', color=co, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='round', zorder=20)

        if green_day is not None and kind == 'cases':
            # variable thickness line (END)
            df = df_orig.copy()
            df = df[green_day:]
            df = df[ ~df['tests_per_hit'].isnull() ]
            x, y = df[[mob_col, Rt_col]].values.T
            tests_per_hit = df['tests_per_hit'].values
            np.place(tests_per_hit, np.isinf(tests_per_hit) | (tests_per_hit > 10000), 10000)
            lwidths = [0.7*np.log(0 + tests_per_hit)]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            for segi, seg in enumerate(segments):
                seg = seg.T
                el = min(1, 0.075 + (lwidths[0][segi]/7)**1.3)
                co = sns.set_hls_values('#77ffaa', l=el)
                ax.plot(seg[0], seg[1], '-', color=co, linewidth=lwidths[0][segi],
                        alpha=1, solid_capstyle='round', zorder=20)

        # dots + thin black
        df = df_orig.copy()
        x, y = df[[mob_col, Rt_col]].values.T
        ax.scatter(x, y, s=0.025, marker='o', facecolor='#000000', alpha=0.5, clip_on=True, zorder=30)
        ax.plot(x, y, linestyle='--', linewidth=0.1, color='#000000', alpha=0.33, zorder=40)

        df = df_orig.copy()
        ax.annotate(s=country, xy=(0.5, 0.88), xycoords='axes fraction', fontsize=9, ha="center",
                    va="center", clip_on=False, zorder=100,
                    path_effects=[pthff.Stroke(linewidth=2, foreground=facecolor), pthff.Normal()])
        if len(df) > 0:
            pop = POPULATION[country]
            total_per_1M = extract_cases_and_deaths(country).loc[last_day][f"total_{kind}"] / pop
            heading = "{:d} {:s}/M".format(int(round(total_per_1M)), kind)
            ax.annotate(s=heading, xy=(0.5, 0.77), xycoords='axes fraction', fontsize=6.5,
                        ha="center", va="center", clip_on=False, zorder=100,
                        path_effects=[pthff.Stroke(linewidth=1.33, foreground=facecolor),
                                      pthff.Normal()])

        adjust_spines(ax, ['left', 'bottom'] if ax.is_first_col() else ['bottom'])
        set_ticks_lengths(ax)

    for ax in fig.axes:
        if ax.is_last_row() and ax.is_last_col():
            ax.set_axis_off()

    legend_fun(fig.axes[-1], base_color)

    fig.tight_layout(w_pad=0.4, h_pad=0.15)
    fn = f"Figure{fig_number}_{last_day.strftime('%b%d')}.pdf"
    fig.savefig(fn)
    print('>>', fn)
    return fig


fig2 = plot_pinworms(TRAJS, eaa_countries, 'cases',  fig_number='2')
fig4 = plot_pinworms(TRAJS, am_countries,  'cases',  fig_number='4')
figS1 = plot_pinworms(TRAJS, eaa_countries, 'deaths', fig_number='S1')
