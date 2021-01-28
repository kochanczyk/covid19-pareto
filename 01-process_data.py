#!/usr/bin/env python3
#pylint: disable = C, R
#pylint: disable = E1101  # no-member (generated-members)
#pylint: disable = C0302  # too-many-lines

"""
  This code features the article

 "Pareto-based evaluation of national responses to COVID-19 pandemic shows
  that saving lives and protecting economy are non-trade-off objectives"

  by Kochanczyk & Lipniacki (Scientific Reports, 2021).

  License: MIT
  Last changes: November 09, 2020
"""

# --------------------------------------------------------------------------------------------------

import re
from operator import itemgetter
from multiprocessing import Pool
import pandas as pd
import numpy as np
import scipy.stats
import dill
import gzip
from shared import *


# -- Contents settings -----------------------------------------------------------------------------

#TMP SNAPSHOT_BASE_URL = 'https://raw.githubusercontent.com/' + \
#TMP                     'kochanczyk/covid19-pareto/master/data/snapshot-20200706/'
SNAPSHOT_BASE_URL = 'data/snapshot-20201109/'  # TMP
OWID_DATA_URL              = SNAPSHOT_BASE_URL + 'owid-covid-data.csv.bz2'
OWID_TESTING_DATA_URL      = SNAPSHOT_BASE_URL + 'covid-testing-all-observations.csv.bz2'
MOBILITY_DATA_URL          = SNAPSHOT_BASE_URL + 'Global_Mobility_Report.csv.bz2'
TRACKING_URL               = SNAPSHOT_BASE_URL + 'daily.csv.bz2'
EXCESS_DEATHS_EUROSTAT_URL = SNAPSHOT_BASE_URL + 'demo_r_mwk_ts_1_Data.csv.bz2'
EXCESS_DEATHS_CDC_URL      = SNAPSHOT_BASE_URL + 'Excess_Deaths_Associated_with_COVID-19.csv.bz2'
GDP_EUROSTAT_URL           = SNAPSHOT_BASE_URL + 'estat_namq_10_gdp--SCA.csv.bz2'


THROWIN_DATES_ = {
    'Spain':          ['2020-04-19', '2020-05-22', '2020-05-25'],
    'France':         ['2020-05-07', '2020-05-29', '2020-06-03'],
    'United Kingdom': ['2020-05-21'],
    'Ireland':        ['2020-05-15'],
    'Portugal':       ['2020-05-03']
}

THROWIN_DATES = {country: list(map(pd.to_datetime, days)) for country, days in THROWIN_DATES_.items()}


# -- Data analysis auxiliary functions -------------------------------------------------------------

def extract_cases_and_deaths(location):
    columns = ['date', 'new_cases', 'total_cases', 'new_deaths', 'total_deaths']
    if is_USA_state(location):
        state_abbrev = STATE_TO_ABBREV[location]
        return TRACKING_DATA.loc[state_abbrev]
    else:
        country = location
        country_indices = OWID_DATA['location'] == country
        return OWID_DATA[country_indices][columns].set_index('date')


def extract_mobility(location):
    if is_USA_state(location):
        df = MOBILITY_DATA[  (MOBILITY_DATA['location'] == 'United States') \
                           & (MOBILITY_DATA['sub_region_1'] == location) \
                           &  MOBILITY_DATA['sub_region_2'].isnull() ].set_index('date')
    else:
        df = MOBILITY_DATA[ (MOBILITY_DATA['location'] == location) \
                           & MOBILITY_DATA['sub_region_1'].isnull() ].set_index('date')
        if 'metro_area' in df.columns:
            df = df[ df['metro_area'].isnull() ]
        assert df['sub_region_1'].isnull().all() and df['sub_region_2'].isnull().all()
    return df


def smoothed_daily_data(location, fix=True):
    daily_ws = [3, 7, 14]
    df = extract_cases_and_deaths(location).copy()

    if fix:
        # general
        for col_new in ('new_cases', 'new_deaths'):
            df[col_new] = df[col_new].fillna(0)
        for col_tot in ('total_cases', 'total_deaths'):
            if pd.isna(df.iloc[0][col_tot]):
                initial_date = df.index[0]
                df.at[initial_date, col_tot] = 0
            df[col_tot] = df[col_tot].ffill()

        # location-specific
        if location in THROWIN_DATES:
            for throwin in THROWIN_DATES[location]:
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

    for k in ('cases', 'deaths'):
        for w in daily_ws:
            df[f"new_{k}{w}"] = df[f"new_{k}"].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
            is_w_even = not (w % 2)
            has_nan_initially = pd.isnull(df.iloc[0][f"new_{k}{w}"])
            if is_w_even and has_nan_initially:
                df.at[df.index[0], f"new_{k}{w}"] = 0

    for col in ('new_cases', 'total_cases', 'new_deaths', 'total_deaths'):
        df[col] = df[col].astype('Int64')

    return df


def calc_Rt(theta, TINY=1e-16):
    if pd.isnull(theta) or theta < TINY:
        return pd.NA
    elif theta == 1:
        return 1
    log2 = np.log(2)
    Td = log2/np.log(theta)
    m, n, sigma, gamma = 6, 1, 1/5.28, 1/3
    Rt = log2/Td*(log2/(Td * m *sigma) + 1)**m / (gamma*(1 - (log2/(Td * n * gamma) + 1)**(-n)))
    return Rt


def insert_epidemic_dynamics(df, timespan_days=14, data_smoothing_window=14):
    half_timespan_days = timespan_days//2
    exponent = (1/(timespan_days - 1))
    for kind in ('cases', 'deaths'):
        values = df[f"new_{kind}{data_smoothing_window}"].values
        thetas = []
        for vi in range(len(values)):
            if vi < half_timespan_days or vi + half_timespan_days >= len(values):
                thetas += [pd.NA]
            else:
                bwd, fwd = values[vi - half_timespan_days], values[vi + half_timespan_days]
                if bwd > 0 and fwd >= 0:
                    theta = (fwd/bwd)**exponent
                    theta = float(theta)
                else:
                    theta = [pd.NA]
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
    def has_day_(dd):          return dd in avg_mo.index
    def is_weekday_(dd):       return dd.dayofweek < 5
    def is_holiday_(cc, dd):   return cc in HOLIDAYS and dd in HOLIDAYS[cc]
    def is_valid_day_(cc, dd): return has_day_(dd) and is_weekday_(dd) and not is_holiday_(cc, dd)

    mo = extract_mobility(location)
    avg_mo = average_mobility_reduction(mo)
    df['mobility'] = avg_mo

    df['mobility_reduction'] = 0
    for day in mo.index:
        if day in df.index and is_valid_day_(location, day):
            df.at[day,'mobility_reduction'] = avg_mo.loc[day]

    for kind in ('cases', 'deaths'):
        distrib = {'cases': INFECTION_TO_REMOVAL, 'deaths': INFECTION_TO_DEATH}[kind]
        df[f"mobility_historical_{kind}"] = pd.NA  # previous values that gave rise to current daily new cases or deaths
        for day in mo.index:
            if day in df.index:
                valid_days_indices = {di for di in range(len(distrib))
                                      if is_valid_day_(location, day - pd.offsets.Day(di))}
                weights     = [distrib[di]
                               for di in valid_days_indices]
                weighted_ms = [distrib[di] * avg_mo.loc[day - pd.offsets.Day(di)]
                               for di in valid_days_indices]
                sum_weights = np.sum(weights)
                df.at[day, f"mobility_historical_{kind}"] = np.sum(weighted_ms)/sum_weights \
                        if sum_weights >= min_sum_weights else pd.NA
    return df


def insert_tests_performed(df, location, interpolate=True, w=7, verbose=False):
    if is_USA_state(location):
        df[f"new_tests{w}"] = df['new_tests'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
        df['tests_per_hit'] = df[f"new_tests{w}"] \
                            / df['new_cases'].rolling(window=w, min_periods=w//2+1, **ROLL_OPTS).mean()
        return df
    else:
        df_test = None
        colnames = ['date', 'Cumulative total']
        endings =  ('tests performed', 'tests performed (CDC) (incl. non-PCR)', 'samples tested',
                    'samples analysed', 'units unclear', 'units unclear (incl. non-PCR)',
                    'people tested', 'people tested (incl. non-PCR)', 'cases tested')

        entities = set(OWID_TESTING_DATA['Entity'])

        location_entities = {}
        for cc, tt in [(e.split(' - ')[0], e.split(' - ')[1]) for e in entities]:
            assert tt in endings
            if cc in location_entities:
                location_entities[cc] = location_entities[cc] + [tt]
            else:
                location_entities[cc] = [tt]
        sel_endings = ['people tested (incl. non-PCR)'] if location == 'Japan' else endings
        for ending in sel_endings:
            ent = f"{location.replace('Czechia', 'Czech Republic')} - {ending}"
            if ent in entities:
                ent_indices = OWID_TESTING_DATA['Entity'] == ent
                if location == 'France':
                    df_fr = OWID_TESTING_DATA[ent_indices][colnames + ['Daily change in cumulative total']]
                    df_fr.at[df_fr.index[0], 'Cumulative total'] = df_fr.iloc[0]['Daily change in cumulative total']
                    for i in range(len(df_fr) - 1):
                        prev_cumulative = df_fr.iloc[i]['Cumulative total']
                        change_in_cumulative = df_fr.iloc[i + 1]['Daily change in cumulative total']
                        df_fr.at[df_fr.index[i + 1], 'Cumulative total'] = prev_cumulative + change_in_cumulative
                    df_pre = df_fr[colnames].set_index('date') \
                            .rename(columns={'Cumulative total': ending})
                else:
                    df_pre = OWID_TESTING_DATA[ent_indices][colnames].set_index('date') \
                            .rename(columns={'Cumulative total': ending})
                if not df_pre[ending].isnull().all():
                    df_test = df_pre if df_test is None else df_test.join(df_pre, how='outer')

        if df_test is None:
            print(f"{location}: missing data on testing")
            df['total_tests'] = np.nan
            df['tests_per_hit'] = np.nan
            return df
        else:
            if verbose:
                print(location, '::',
                      df_test.index[ 0].strftime('%Y: %B, %d'), '--',
                      df_test.index[-1].strftime('%B, %d'), '::', ', '.join(list(df_test.columns)))

        if len(df_test.columns) == 1:
            df_test.rename(columns=lambda colnm: re.sub(r'^.*$', 'total_tests', colnm), inplace=True)
        else:
            df_test['total_tests'] = np.nan
            df_test['test_type'] = '?'
            for ending in endings:
                if ending not in df_test.columns: continue
                for day in df_test.index:
                    if np.isnan(df_test.loc[day]['total_tests']) and not np.isnan(df_test.loc[day][ending]):
                        df_test.at[day, 'total_tests'] = df_test.loc[day][ending]
                        df_test.at[day, 'test_type'] = ending
            if verbose:
                for ending in endings:
                    if ending not in df_test.columns: continue
                    df_sub = df_test[ df_test['test_type'] == ending ][ending].dropna()
                    if len(df_sub):
                        print(' '*len(location), '::',
                            df_sub.index[ 0].strftime('%Y: %B, %d'), '--',
                            df_sub.index[-1].strftime('%B, %d'), '::', ending)

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


def check_gaps(location, traj):
    missing = []
    dt = traj['new_deaths'].index[-1] - traj['new_deaths'].index[0]
    if dt.days != len(traj['new_deaths'].index) - 1:
        for i in range(len(traj.index) - 1):
            since, until = traj.index[i], traj.index[i + 1]
            inter_days = (until - since).days
            if inter_days > 1:
                gap = inter_days - 1
                if gap == 1:
                    timespan_s = f"{(since + pd.offsets.Day(1)).strftime('%B %d')}"
                else:
                    timespan_s = f"{(since + pd.offsets.Day(1)).strftime('%B %d')}--" \
                                 f"{(until - pd.offsets.Day(1)).strftime('%B %d')}"
                for i in range(gap):
                    day = since + pd.offsets.Day(1 + i)
                    if day < FINAL_DAY:
                        missing += []
                print(f"{location}: missing {gap} day{'s' if gap > 1 else ''}: {timespan_s}")
    return missing


def check_mobility(location, trajectory):
    missing = []
    nan_blocks = []
    in_nan_block = False
    for index, value in trajectory[['mobility']].iterrows():
        if pd.isnull(float(value)):
            if index < FINAL_DAY:
                missing += [index]
            if not in_nan_block:
                in_nan_block = True
                nan_blocks.append([index])
            else:
                nan_blocks[-1].append(index)
        else:
            if in_nan_block:
                in_nan_block = False
    for nan_block in nan_blocks:
        since, until = nan_block[0], nan_block[-1]
        if since != trajectory.index[0] and until != trajectory.index[-1]:
            timespan_s = f"{since.strftime('%B %d')}--" \
                         f"{until.strftime('%B %d')}"
            print(f"{location}: missing mobility: {timespan_s}")
    return missing


# --------------------------------------------------------------------------------------------------

# https://ec.europa.eu/eurostat/statistics-explained/images/d/da/Weekly_deaths_15_10_2020-update.xlsx
# which are source data for:
# https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Weekly_death_statistics

def read_excess_deaths_eurostat():
    d = pd.read_csv(EXCESS_DEATHS_EUROSTAT_URL) \
          .drop(columns=['UNIT', 'Flag and Footnotes', 'SEX'])
    d.loc[ d['Value']==':', 'Value'] = pd.NA
    d['Value'] = d['Value'].map(lambda v: int(v.replace(',', '')) if type(v)==str else v)
    d = d[ d['GEO'] != 'Georgia' ]
    weeks = [f"W{i:0>2d}" for i in range(1, 13+13+1)]
    years = list(map(str, range(2016, 2020)))
    excess_deaths = {}
    for loc, dd in d.groupby('GEO'):
        dd.set_index('TIME', inplace=True)
        ddd = {wk: dd.loc[ [f"{yr}{wk}" for yr in years] ]['Value'].mean()
               for wk in weeks}
        ddd = pd.DataFrame.from_dict(ddd, orient='index', columns=['Average deaths'])
        ddd['2020 deaths'] = [dd.loc[ [f"2020{wk}" for yr in years] ]['Value'].mean()
                              for wk in weeks]
        ddd['2020 excess deaths'] = ddd['2020 deaths'] - ddd['Average deaths']
        loc = loc.replace('Germany (until 1990 former territory of the FRG)', 'Germany')
        excess_deaths[loc] = (ddd['2020 deaths'] - ddd['Average deaths']).sum()
    return excess_deaths


def read_excess_deaths_cdc():
    d = pd.read_csv(EXCESS_DEATHS_CDC_URL, parse_dates=['Week Ending Date'])
    d = d[ (d['Type'] == 'Predicted (weighted)') & (d['Outcome']=='All causes') ]
    d = d[ (pd.to_datetime('2020-01-01') <= d['Week Ending Date']) & \
                                           (d['Week Ending Date'] <= pd.to_datetime('2020-07-04')) ]
    d = d.groupby('State').sum()
    return dict(d['Observed Number'] - d['Average Expected Count'])


def estimate_GDP_2020H1():
    d = pd.read_csv(GDP_EUROSTAT_URL, sep=r'[\t,]', engine='python')
    d.rename(columns=lambda colname: colname.strip(), inplace=True)
    d.rename(columns={'geo\TIME_PERIOD': 'country'}, inplace=True)
    d['country'].replace(EUROPEAN_COUNTRY_CODES, inplace=True)
    d = d[ d['country'].apply(lambda geo: geo not in 'EA EA12 EA19 EU15 EU27_2020 EU28'.split()) ]

    to_number = lambda s: pd.NA if s==':' else s if type(s)==float else float(s.split()[0])
    for colname in d.columns:
        is_q_colname = re.match(re.compile(r'^20[0-9][0-9]-Q[0-9]$'), colname) is not None
        if is_q_colname:
            d.at[:, colname] = d[colname].apply(to_number)

    GDP_2020H1YOY_Eurostat = {}
    for country in d['country'].values:
        dd = d[ d['country'] == country ]
        gdp_2019_H1 = float(dd['2019-Q1'] + dd['2019-Q2'])
        gdp_2020_H1 = float(dd['2020-Q1'] + dd['2020-Q2'])
        GDP_2020H1YOY_Eurostat[country] = (gdp_2020_H1 - gdp_2019_H1) / gdp_2019_H1 * 100


    # "2020 Q1+Q2" / "2019 Q1+Q2" - 1:

    # USD inflation rate as average in 12 months of 2019 H2 and 2020 H1
    # [https://www.usinflationcalculator.com/inflation/current-inflation-rates]
    USD_infl_rate = np.mean([1.8, 1.7, 1.7, 1.8, 2.1, 2.3, 1.8, 2.5, 2.3, 1.5, 0.3, 0.1, 0.6])/100
    GDP_2020H1YOY_BEA = {
        'California':     ((2893054 + 3189703)*(1 - USD_infl_rate) / (3119174 + 3063191) - 1)*100,
        'Texas':          ((1628185 + 1818394)*(1 - USD_infl_rate) / (1835576 + 1827426) - 1)*100,
        'Florida':        ((1026676 + 1121367)*(1 - USD_infl_rate) / (1098679 + 1087641) - 1)*100,
        'New York':       ((1587879 + 1778240)*(1 - USD_infl_rate) / (1771545 + 1746191) - 1)*100,
        'Pennsylvania':   (( 723830 +  808937)*(1 - USD_infl_rate) / ( 805933 +  797988) - 1)*100,
        'Illinois':       (( 807383 +  884447)*(1 - USD_infl_rate) / ( 880445 +  878173) - 1)*100,
        'Ohio':           (( 626275 +  696274)*(1 - USD_infl_rate) / ( 691885 +  688012) - 1)*100,
        'Georgia':        (( 580732 +  631346)*(1 - USD_infl_rate) / ( 622814 +  616437) - 1)*100,
        'North Carolina': (( 546776 +  600631)*(1 - USD_infl_rate) / ( 588477 +  581020) - 1)*100,
        'Michigan':       (( 475494 +  535153)*(1 - USD_infl_rate) / ( 533727 +  529751) - 1)*100,
    }

    GDP_2020H1YOY_OECD = {
        'Canada':         ((106.3 +  94.0) / (107.2 + 108.1) - 1)*100,
        'Japan':          ((101.9 +  93.8) / (103.9 + 104.3) - 1)*100,
        'South Korea':    ((111.7 + 108.1) / (110.1 + 111.2) - 1)*100,
    }

    # [https://www.dgbas.gov.tw/ct.asp?xItem=45796&ctNode=3339&mp=1]
    GDP_2020H1YOY_DGBAS = {
        'Taiwan':         ((4820427 + 4770934) / (4732831 + 4754127) - 1)*100,
    }


    GDP_2020H1YOY = {
        **GDP_2020H1YOY_Eurostat,
        **GDP_2020H1YOY_BEA,
        **GDP_2020H1YOY_OECD,
        **GDP_2020H1YOY_DGBAS
    }

    return GDP_2020H1YOY


# ==================================================================================================


OWID_DATA = pd.read_csv(OWID_DATA_URL, parse_dates=['date']) \
              .replace({'Czech Republic': 'Czechia'})
OWID_DATA['date'] = OWID_DATA['date'].apply(pd.to_datetime)

MOBILITY_DATA = pd.read_csv(MOBILITY_DATA_URL, parse_dates=['date'], low_memory=False) \
                  .rename(columns=lambda colnm: re.sub('_percent_change_from_baseline$', '', colnm)) \
                  .rename(columns={'country_region': 'location'})

OWID_TESTING_DATA = pd.read_csv(OWID_TESTING_DATA_URL, parse_dates=['Date']) \
                      .rename(columns={'Date': 'date'})

TRACKING_DATA = pd.read_csv(TRACKING_URL, parse_dates=['date'])[::-1].set_index(['state', 'date'])
TRACKING_DATA['new_tests'] = TRACKING_DATA['negativeIncrease'] + TRACKING_DATA['positiveIncrease']
TRACKING_DATA['total_tests'] = TRACKING_DATA['negative'] + TRACKING_DATA['positive']
TRACKING_DATA.rename(columns={'positive': 'total_cases', 
                              'positiveIncrease': 'new_cases',
                              'death': 'total_deaths', 
                              'deathIncrease': 'new_deaths'}, inplace=True)


LOCATIONS_FLAT = [c for cs in LOCATIONS.values() for c in cs]
for c in LOCATIONS_FLAT:
    if c in LOCATIONS['USA']:
        assert STATE_TO_ABBREV[c] in TRACKING_DATA.index.get_level_values(0).unique()
    else:
        assert len(OWID_DATA[ OWID_DATA['location'] == c ])
        assert c in set(MOBILITY_DATA['location'])


SELECTED_LOCATIONS = [ c for part in ['Europe', 'USA'] for c in LOCATIONS[part] 
                       if population(c) >= MIN_POPULATION_M ] \
                   + ['Canada', 'Taiwan', 'Japan', 'South Korea']
TRAJS = dict(Pool().map(process_location, SELECTED_LOCATIONS))


NO_LARGE_GAPS_SINCE = pd.to_datetime('2020-03-01')
MISSING_DAYS = {}
locations_to_remove = []
for location, trajectory in TRAJS.items():
    missG = check_gaps    (location, trajectory)
    missM = check_mobility(location, trajectory)
    MISSING_DAYS[location] = set(missG).union(set(missM))
    if len([day for day in MISSING_DAYS[location] if day >= NO_LARGE_GAPS_SINCE]) > 7:
        locations_to_remove.append(location)
for location in locations_to_remove:
    del TRAJS[location]
    SELECTED_LOCATIONS.remove(location)
    print(f"NOTE: {location} removed due to missing data.")


excess_deaths_eurostat = read_excess_deaths_eurostat()
excess_deaths_cdc = read_excess_deaths_cdc()
EXCESS_DEATHS = {**excess_deaths_eurostat, **excess_deaths_cdc}

GDP_2020H1 = estimate_GDP_2020H1()


#dill.dump_session('session.dill')

with gzip.open('processed_data.dill.gz', 'wb') as f:
    pack = [TRAJS, SELECTED_LOCATIONS, FINAL_DAY, MISSING_DAYS, EXCESS_DEATHS, GDP_2020H1]
    dill.dump(pack, f)
