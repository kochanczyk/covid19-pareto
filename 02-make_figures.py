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
import seaborn as sns
import numpy as np
import scipy.stats
import statsmodels.stats.weightstats as wstats
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.ticker as tckr
import matplotlib.patheffects as pthff
from colorsys import rgb_to_hls
from pandas.plotting import register_matplotlib_converters
import locale
import dill
import gzip
from shared import *


register_matplotlib_converters()

locate_set = False
try:
    locale.setlocale(locale.LC_TIME, 'en_US')
    locale.setlocale(locale.LC_ALL,  'en_US')
    locate_set = True
except:
    try:
        locale.setlocale(locale.LC_TIME, 'en_US.utf8')
        locale.setlocale(locale.LC_ALL,  'en_US.utf8')
        locate_set = True
    except:
        locale.setlocale(locale.LC_TIME, 'POSIX')
        locale.setlocale(locale.LC_ALL,  'POSIX')
if not locate_set:
    print('Warning: US English locale could not be set. Check tick labels in generated figures.')


# -- Shared plot settings --------------------------------------------------------------------------

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.pad'] = 1.67
plt.rcParams['ytick.major.pad'] = 1.33
plt.rc('font', size=8, family='sans-serif')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'''\usepackage{cmbright}''')



# -- Plotting auxiliary functions ------------------------------------------------------------------

# manual tweaks:
OUT_OF_FRONT = ['Greece', 'Hungary', 'Canada', 'Netherlands', 'Czechia']

# colors:
SNAIL_GREEN, SNAIL_NONGREEN, SNAIL_ORANGE = '#77ffaa', '#aabbdd', '#885500'
ANNOT_COLOR = '#777777'

def color_of(country, dull_color=(0.15, 0.15, 0.15)):
    colors = {
        'Austria':        plt.cm.tab10(6),
        'Belgium':        plt.cm.tab10(5),
        'Bulgaria':       plt.cm.tab10(2),
        'Croatia':        (0.50, 0.55, 0.00),
        'Czechia':        plt.cm.tab10(4),
        'Denmark':        (0.85, 0.20, 0.00),
        'Finland':        plt.cm.tab10(9),
        'France':         (0.95, 0.25, 0.75),
        'Germany':        (0.55, 0.25, 0.70),
        'Hungary':        (0.35, 0.35, 0.35),
        'Greece':         (0.45, 0.75, 1.00),
        'Italy':          plt.cm.tab10(2),
        'Netherlands':    (0.88, 0.50, 0.00),
        'Norway':         plt.cm.tab10(0),
        'Poland':         (0.15, 0.65, 1.00),
        'Portugal':       (0.95, 0.65, 0.00),
        'Romania':        plt.cm.tab10(8),
        'Russia':         (0.80, 0.45, 0.15),
        'Slovakia':       (0.25, 0.90, 0.50),
        'Slovenia':       plt.cm.tab10(1),
        'Spain':          plt.cm.tab10(3),
        'Sweden':         (0.10, 0.20, 0.90),
        'Switzerland':    (1.00, 0.05, 0.05),
        'United Kingdom': (0.20, 0.00, 0.99),

        'Japan':          (0.9,  0.00, 0.00),
        'South Korea':    (0.70, 0.60, 0.65),
        'Taiwan':         (0.10, 0.80, 0.00),

        'California':     (0.90, 0.70, 0.00),
        'Canada':         (0.00, 0.45, 0.80),
        'Florida':        (0.95, 0.40, 0.00),
        'Georgia':        (0.80, 0.10, 0.60),
        'Illinois':       (0.75, 0.50, 0.00),
        'Michigan':       (0.05, 0.50, 0.15),
        'North Carolina': (0.10, 0.00, 0.95),
        'New York':       (0.60, 0.30, 0.00),
        'Ohio':           (0.65, 0.00, 0.00),
        'Pennsylvania':   (0.20, 0.25, 1.00),
        'Texas':          (0.35, 0.40, 0.40),

        'Argentina':      (0.30, 0.75, 1.00),
        'Bolivia':        (0.20, 0.65, 0.00),
        'Brazil':         (0.00, 0.70, 0.20),
        'Chile':          (0.65, 0.15, 0.00),
        'Colombia':       (0.00, 0.10, 0.65),
        'Ecuador':        (0.65, 0.65, 0.00),
        'Mexico':         (0.00, 0.50, 0.60),
        'Peru':           (0.75, 0.50, 0.25),
    }
    if country in colors.keys():
        return colors[country]
    else:
        return dull_color


def correlations(values, weights):
    rho = scipy.stats.pearsonr(values[:,0], values[:,1])[0]
    wrho = wstats.DescrStatsW(values, weights=weights).corrcoef[0][1]
    return (rho, wrho)


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


def darken(color, scale=0.5):
    lightness = min(1, rgb_to_hls(*color[0:3])[1] * scale)
    return sns.set_hls_values(color=color, h=None, l=lightness, s=None)


def pareto_front(data, optima=True):
    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=not optima)  # x-ascending
    front = [ sorted_data[0][2] ]
    cutoff = sorted_data[0][1]
    for sd in sorted_data[1:]:
        if (optima and sd[1] < cutoff) or (not optima and sd[1] > cutoff):
            front += [sd[2]]
            cutoff = sd[1]
    return front


def put_final_dot(ax, location, x, y, is_extra_country=False, is_tail_shown=False,
                  show_population_halo=False, label_shifting='A', italic=False):
    label_shifts = {
        'Denmark':        (940, 1.0  ),
        'Norway':         ( 20, 0.88 ),
        'South Korea':    ( 52, 0.59 ),
        'Portugal':       (  0, 0.97 ),
        'Bulgaria':       (830, 0.994),
        'Switzerland':    ( 80, 0.92 ),
        'Ohio':           ( 40, 1.014),
        'Michigan':       (800, 1.018),
        'Florida':        (  0, 0.987),
        'Illinois':       ( 90, 1.016),
        'North Carolina': (-10, 0.97 ),
        'Pennsylvania':   (  0, 0.999),
        'Georgia':        (825, 0.991)
    } if label_shifting == 'A' else {}
    if show_population_halo:
        marker_size = 3.5
        diameter = np.sqrt(population(location)) * 3
        light_color = color_of(location)
        ax.plot([x], [y], '-.', marker='8' if is_extra_country else 'o',
                linewidth=1, markersize=diameter, markeredgewidth=0, alpha=0.2, clip_on=False,
                color=light_color, markerfacecolor=light_color)
    else:
        marker_size = 6

    ax.plot([x], [y], '-.', marker='8' if is_extra_country else 'o',
            linewidth=1, markersize=marker_size, markeredgewidth=0, alpha=0.8, clip_on=False,
            color=color_of(location), markerfacecolor=color_of(location))
    loc = location.replace('United Kingdom', 'UK')
    if italic:
        loc = r'\textit{' + loc + r'}'
    if label_shifting == 'A':
        ax.annotate(loc, xycoords='data',
                    xy=(x + 65    - (0 if location not in label_shifts else label_shifts[location][0]),
                        y**0.9999 * (1 if location not in label_shifts else label_shifts[location][1])),
                    color=sns.set_hls_values(color_of(location), l=0.3), clip_on=False)
    else:
        ax.annotate(loc, xycoords='data',
                    xy=(x + 0.13,
                        y + 0.04),
                    color=sns.set_hls_values(color_of(location), l=0.3), clip_on=False)


def jointly_trimmed_trajs(trajs, locations, cols, force_end=None, skipped=None, cleanup=True,
                          verbose=False):
    assert len(cols) == 2
    col1, col2 = cols
    days_of_last_available_data = set()
    for country in locations:
        if skipped and country in skipped:
            continue
        df = trajs[country]
        df_sel = df[ ~df[col1].isnull() & ~df[col2].isnull() ]
        last_day = df_sel.iloc[-1].name
        days_of_last_available_data.add(last_day)
        if verbose:
            print(country, last_day.strftime('%b%d'))
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
    for country in locations:
        df = trajs[country].loc[:day_of_last_available_data]
        edited_trajs[country] = df[ ~df[col1].isnull() & ~df[col2].isnull() ] if cleanup else df

    return day_of_last_available_data, edited_trajs


def extract_cumulative_immobilization_and_deaths(trajectories, country, interval):
    trajectory = trajectories[country]
    immobi = -trajectory[['mobility_reduction']]
    deaths =  trajectory[['new_deaths']].astype('Float64')
    ppl = population(country)
    if interval == 'monthly':
        immobi = immobi.cumsum().groupby(pd.Grouper(freq='M')).nth(0)
        deaths = deaths.cumsum().groupby(pd.Grouper(freq='M')).nth(0) / ppl
        df = immobi.join(deaths).rename(columns={
                'mobility_reduction':  f"immobilization_cumul_{country}",
                'new_deaths': f"new_deaths_cumul_per_1M_{country}"})
        ii = df.index
        df.index = [i.replace(day=1) for i in ii]
        return df
    elif interval == 'weekly':
        immobi = immobi.resample('W').sum().cumsum()
        deaths = deaths.resample('W').sum().cumsum() / ppl
        df = immobi.join(deaths).rename(columns={
                'mobility_reduction':  f"immobilization_cumul_{country}",
                'new_deaths': f"new_deaths_cumul_per_1M_{country}"})
        return df
    elif interval == 'daily':
        immobi = immobi.cumsum()
        deaths = deaths.cumsum() / ppl
        df = immobi.join(deaths).rename(columns={
                'mobility_reduction':  f"immobilization_cumul_{country}",
                'new_deaths': f"new_deaths_cumul_per_1M_{country}"})
        return df



def make_sqrt_deaths_yaxis(ax, ymax=40, sep=5):
    ax.set_ylim((0, ymax))
    ticks = list(range(0, ymax + sep, sep))
    ax.set_yticks(ticks)
    ax.set_yticklabels(['0'] + [r'$\sqrt{' + str(t**2) + '}$' for t in ticks[1:]])


def plot_cumulative_immobilization_and_deaths(trajectories, locations, final_day, show_fronts,
                                              show_tail, show_corr_history, show_population_halo,
                                              fig_name='X', scale_deaths=np.sqrt):

    def draw_pareto_fronts_(ax, finals, n_fronts, optimal):
        fronts = []
        for i in range(n_fronts):
            fronts_locations = [__ for _ in fronts for __ in _]
            finals_remaining = [(*im_de, loc) for loc, im_de in finals.items()
                                if loc not in fronts_locations and loc not in OUT_OF_FRONT]
            front = pareto_front(finals_remaining, optimal)
            fronts.append(front)

        for front_i, front in enumerate(fronts):
            color = sns.set_hls_values('gray', l=0.1 + 0.04*(max(0, front_i - 1*optimal))) # TMP: was 0.15+0.1*
            front_coords = np.array([finals[loc] for loc in front]).T
            if len(front_coords.T) > 1:
                ax.plot(*front_coords, ':' if optimal else '--', c=color, alpha=0.8,
                        linewidth=1.1 if optimal else 0.8)
            else:
                if optimal:
                    front_coords = [[front_coords[0][0] + 0.707*180  + 180*np.cos((180 + i)/360*2*3.14159),
                                     front_coords[1][0] + 0.8        + 1.2*np.sin((180 + i)/360*2*3.14159)]
                                    for i in range(0, 91, 10)]
                else:
                    front_coords = [[front_coords[0][0] - 0.707*180  + 180*np.cos((180 + i)/360*2*3.14159),
                                     front_coords[1][0] - 0.8        + 1.2*np.sin((180 + i)/360*2*3.14159)]
                                    for i in range(180+0, 180+91, 10)]
                ax.plot(*np.array(front_coords).T, ':' if optimal else '--', c=color, alpha=0.8,
                        linewidth=1.1 if optimal else 0.8, clip_on=False)


    def make_subplot_(ax, trajs, locations, final_day, show_fronts, panel_letter=None):
        adjust_spines(ax, ['left', 'bottom'], left_shift=10)
        ax.set_xlim((0, 8e3))
        ax.set_xlabel(r'Cumulative lockdown')
        ax.set_ylabel(r'$\sqrt{\textrm{\sf Cumulative deaths/M}}$')
        make_sqrt_deaths_yaxis(ax)

        # plot "flares" (tails are optional)
        finals = {}
        for loc in locations:
            im, de = extract_cumulative_immobilization_and_deaths(trajs, loc, 'monthly').values.T
            de = scale_deaths(de)
            put_final_dot(ax, loc, im[-1], de[-1], show_population_halo=show_population_halo)
            if show_tail:
                color = color_of(loc)
                darker_color = darken(color_of(loc))
                alpha = 0.7
                ax.plot(im, de, '-', linewidth=0.8, alpha=alpha, color=color)
                for i in range(1, len(im)):
                    m, ms = [('s', 1.7), ('D', 1.55), ('p', 2.2)][i % 3]
                    ax.plot(im[-1 - i], de[-1 - i], '.', marker=m, markersize=ms,
                            fillstyle=None, markeredgewidth=0.33, markerfacecolor=darken(color, 0.9),
                            markeredgecolor=darker_color, alpha=alpha)
                ax.plot(im[-1], de[-1], '.', marker='o', markersize=1., markeredgewidth=0,
                        markerfacecolor=darker_color, alpha=alpha)
            finals[loc] = (im[-1], de[-1])

        if show_fronts:
            draw_pareto_fronts_(ax, finals, n_fronts=3+2, optimal=True)
            draw_pareto_fronts_(ax, finals, n_fronts=2, optimal=False)

        # annotation: last day
        ax.annotate(str('Date:' if show_corr_history else 'Last day:') + \
                    f" {final_day.strftime('%B %d, %Y')}", xy=(0.0, 1.01), xycoords='axes fraction',
                    color=ANNOT_COLOR)

        # annotation: correlation coefficients
        values = np.array(list(finals.values()))
        weights = np.array([population(loc) for loc in finals.keys()])
        rho, wrho = correlations(values, weights)
        ax.annotate(r'Correlation:',
                    xy=(0.0, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
        ax.annotate(r"(non-weighted) Pearson's $\rho$ = " + f"{rho:.2f}",
                    xy=(0.16 - 0.03*show_tail, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
        ax.annotate(r"population-weighted Pearson's $\rho$ = " + f"{wrho:.2f}",
                    xy=(0.16 - 0.03*show_tail, 0.94), xycoords='axes fraction', color=ANNOT_COLOR)

        # export coordinates
        if panel_letter is not None:
            csv_fn = f"Figure{fig_name}{panel_letter}.csv"
            np.savetxt(csv_fn, values, header='lockdown,sqrt_deaths', delimiter=',')


    cols = ['mobility', 'new_deaths']

    # set up the figure
    if show_corr_history:

        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        for i, fday in enumerate(final_day):
            last_avail_day, trajs = jointly_trimmed_trajs(trajectories, locations, cols, force_end=fday)
            assert fday <= last_avail_day
            panel_letter = chr(ord('A') + i)
            make_subplot_(axes[i], trajs, locations, fday, show_fronts=show_fronts and i>0,
                          panel_letter=panel_letter)
            axes[i].annotate(r'\large\textbf{' + panel_letter + r'}',
                            xy=(-0.175, 1.04), xycoords='axes fraction', clip_on=False)

        ax = axes[1].inset_axes([0.92, 0.09, 0.45, 0.2])
        adjust_spines(ax, ['left', 'bottom'], left_shift=7)
        ax.annotate(r'\large\textbf{C}', xy=(-0.275, 1.06), xycoords='axes fraction', clip_on=False)
        x, y1, y2 = [], [], []
        for i in range(9):
            points, weights = [], []
            for loc in locations:
                im_de = extract_cumulative_immobilization_and_deaths(trajs, loc, 'monthly').iloc[-1 - i]
                points.append([im_de[0], scale_deaths(im_de[1])])
                weights.append(population(loc))
            points = np.array(points)
            rho, wrho = correlations(points, weights)
            x.append(im_de.name)
            y1.append(rho)
            y2.append(wrho)
        ax.xaxis.set_major_formatter(dts.DateFormatter('%b'))  # %d
        ax.yaxis.set_major_locator(tckr.MultipleLocator(0.1))
        ax.plot(x, y2, '.-', linestyle='dotted', linewidth=0.5, color='#333333', markersize=7,
                markerfacecolor='#00000000', markeredgecolor='black', markeredgewidth=0.5,
                label=r'population-weighted $\rho$')
        ax.plot(x, y1, '.-', linestyle='dashed', linewidth=0.5, color='#333333', markersize=5.5,
                label=r'non-weighted $\rho$')
        ax.set_ylim((0.5, 0.9))
        ax.set_xlabel(r'First days of months of 2020')
        ax.set_ylabel(r"Pearson's $\rho$")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.48), fancybox=False, fontsize=6.75)
        for item in (ax.xaxis.label, ax.yaxis.label):                    item.set_fontsize(7.00)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):     label.set_fontsize(6.25)

    else:
        last_avail_day, trajs = jointly_trimmed_trajs(trajectories, locations, cols, force_end=final_day)
        assert final_day <= last_avail_day
        fig, axes = plt.subplots(ncols=1, figsize=(6, 5))
        make_subplot_(axes, trajs, locations, final_day, show_fronts=False, panel_letter='_')

    # export
    fig.tight_layout()
    fn = f"Figure{fig_name}.pdf"  # _{last_day.strftime('%b%d')}
    fig.savefig(fn)
    print(f"Saved figure file {fn}.")
    return fig


def put_legend_cases(ax_leg, thr_weekly_cases_per_1M):

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
        color = sns.set_hls_values(SNAIL_NONGREEN, l=0.15 + (lwidths[0][segi] - 0.)/8)
        ax_leg.plot(seg[0]+0.05, seg[1], '-', color=color, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='butt', zorder=20, clip_on=False)

    # variable thickness line (END)
    points = np.array([x, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    for segi, seg in enumerate(segments):
        seg = seg.T
        el = min(1, 0.075 + ((lwidths[0][segi] - 0.)/7)**1.3)
        co = sns.set_hls_values(SNAIL_GREEN, l=el)
        ax_leg.plot(seg[0]+0.05, seg[1], '-', color=co, linewidth=lwidths[0][segi],
                    alpha=1, solid_capstyle='butt', zorder=20, clip_on=False)

    # dots + thin black
    for y in [y1, y2, y3]:
        xx, yy = x[:-1], y[:-1]
        ax_leg.scatter(xx + 0.5, yy, s=0.025, marker='o', facecolor='#000000', alpha=0.5,
                       clip_on=False, zorder=30)
        ax_leg.plot(xx + 0.5, yy, linestyle='--', linewidth=0.1, color='#000000', alpha=0.33,
                    clip_on=False, zorder=40)

    ax_leg.annotate(text=r'Tests per case:', xy=(0.5, 0.84), xycoords='axes fraction', fontsize=8,
                    ha="center", va="center")
    ax_leg.annotate(text=r'when \textbf{$>$ ' + str(thr_weekly_cases_per_1M) + r'} '
                      r'new cases /week /M', xy=(0.5, 0.62-0.09),
                    xycoords='axes fraction', fontsize=6.5, ha="center", va="center")
    ax_leg.annotate(text=r'when \textbf{$<$ ' + str(thr_weekly_cases_per_1M) + '} '
                      r'new cases /week /M', xy=(0.5, 0.31-0.09),
                    xycoords='axes fraction', fontsize=6.5, ha="center", va="center")
    ax_leg.annotate(text=r'no data on testing', xy=(0.5, 0.055), xycoords='axes fraction',
                    fontsize=6.5, ha="center", va="center")

    for vi, v in enumerate(z):
        for y in [y1, y2]:
            extra_shift = -0.08 if v in [100, 300, 1000] else 0
            ax_leg.annotate(text=f"{v}"[::-1].replace('000', 'k')[::-1], color='black',
                            xy=(x[vi]+extra_shift + 0.5, y[vi]+0.05+0.005*vi), xycoords='data',
                            fontsize=5.75, ha="center", va="center", zorder=30, clip_on=False)


def put_legend_deaths(ax_leg):

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
        color = sns.set_hls_values(SNAIL_ORANGE, l=el)
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


def plot_R_vs_mobility_reduction(trajs, locations, final_day, missing_days, fig_name, kind='cases',
                thr_weekly_cases_per_1M=20):

    assert kind in ('cases', 'deaths')

    trajs_orig = trajs.copy()

    low_mortality_locations = ['Taiwan', 'Slovakia', 'New Zealand']
    mob_col, Rt_col = f"mobility_historical_{kind}", f"Rt_{kind}"
    last_day, trajs_trimmed = jointly_trimmed_trajs(trajs, locations, [mob_col, Rt_col],
                                                    force_end=final_day,
                                                    skipped=low_mortality_locations)

    def by_per_capita(cc):
        if kind == 'cases':
            assert last_day in trajs[cc].index, \
                    print(f"Day {last_day} not available for {cc} that ends on",
                          trajs[cc].tail(1).index)
            return trajs[cc].loc[last_day, f"total_{kind}"] / population(cc) + 1e6*is_USA_state(cc)
        elif kind == 'deaths':
            if cc in low_mortality_locations:
                return trajs[cc].loc[last_day, f"total_{kind}"] / 1e9 + 1e6*is_USA_state(cc)
            else:
                return trajs[cc].loc[last_day, f"total_{kind}"] / population(cc) + 1e6*is_USA_state(cc)
    locations = sorted(locations, key=by_per_capita, reverse=True)

    facecolor = '#f8f6f4'
    ncols = 6
    nrows = (len(locations))//ncols + 1
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8/5*ncols, 8/6*nrows))

    for ci, country in enumerate(locations):
        ax = fig.axes[ci]
        ax.set_facecolor(facecolor)

        # PLOT: deaths in low-mortality locations
        if kind == 'deaths' and country in low_mortality_locations:
            ax.annotate(s=country, xy=(0.5, 0.88), xycoords='axes fraction', fontsize=9,
                        color='#666666', ha="center", va="center", clip_on=False, zorder=100)
            total = trajs_orig[country].loc[last_day, f"total_{kind}"]
            ax.annotate(s="{:d} {:s} in total".format(int(round(total)), kind),
                        xy=(0.5, 0.77), xycoords='axes fraction', fontsize=6.5, color='#666666',
                        ha="center", va="center", clip_on=False, zorder=100)
            ax.annotate(s="(plot not shown)",
                        xy=(0.5, 0.67), xycoords='axes fraction', fontsize=6.5, color='#666666',
                        ha="center", va="center", clip_on=False, zorder=100)
            adjust_spines(ax, ['left', 'bottom'] if ax.is_first_col() else ['bottom'])
            ax.set_xticks(())
            continue

        # PLOT: X-axis
        row_i = ci//ncols
        if row_i == nrows-1:
            ax.set_xlabel('Mobility', labelpad=-1)
        ax.set_xlim((-100, 0))
        ax.set_xticks((-100, 0))
       #ax.xaxis.set_major_formatter(tckr.PercentFormatter(decimals=0))
        ax.set_xticklabels((r'$-100\%$', r'$0\%$'))

        # PLOT: Y-axis
        if ax.is_first_col():
            ax.set_ylabel(r'$R$')
        ax.set_ylim((0, 4))
        ax.yaxis.set_major_locator(tckr.MultipleLocator(1))
        ax.axhline(1, linestyle='--', linewidth=0.5, color='#666666')

        # DATA
        df = trajs_trimmed[country].copy()

        # DATA: begin each trajectory since 100 cumulative cases
        min_cumul = 100
        above_min_cumul_indices = df['total_cases'] >= min_cumul  # cases even if kind == 'deaths'
        df = df[above_min_cumul_indices]

        # DATA: nullify missing days to obtain visual discontinuities
        for missing_day in missing_days[country]:
            if df.index[0] <= missing_day and missing_day <= FINAL_DAY:
                df.at[missing_day,mob_col] = np.nan  # cannot be pd.NA because used in mpl.plot
                df.at[missing_day, Rt_col] = np.nan  # cannot be pd.NA because used in mpl.plot
        df.sort_index(inplace=True)


        if kind == 'cases':  # ==---

            # PLOT: pink tracer line
            ax.plot(*df[[mob_col, Rt_col]].values.T, linestyle='-', linewidth=0.75, alpha=1,
                    solid_capstyle='round', color='#ffaaee', clip_on=True, zorder=10)

            # DATA: partition trajectory into temporally-ordered stretches
            df_freq = df[f"new_{kind}"].ffill().rolling(window=7, min_periods=7, **ROLL_OPTS).sum()\
                     / population(country)
            assert len(df_freq) == len(df)
            green_indices    = df[df_freq <  thr_weekly_cases_per_1M].index
            nongreen_indices = df[df_freq >= thr_weekly_cases_per_1M].index
            green_stretches, nongreen_stretches = [], []
            last_index_is_green = None
            for index, value in df.iterrows():
                if index in green_indices:
                    if last_index_is_green is None or last_index_is_green == False:
                        green_stretches += [ [index] ]
                    elif last_index_is_green == True:
                        green_stretches[-1] += [index]
                    last_index_is_green = True
                elif index in nongreen_indices:
                    if last_index_is_green is None or last_index_is_green == True:
                        if green_stretches:
                            green_stretches[-1] += [index]  # extra point for smooth joins
                        nongreen_stretches += [ [index] ]
                    elif last_index_is_green == False:
                        nongreen_stretches[-1] += [index]
                    last_index_is_green = False
            stretches =  [( g, SNAIL_GREEN   ) for  g in    green_stretches] \
                       + [(ng, SNAIL_NONGREEN) for ng in nongreen_stretches]
            def by_first_day(cs):  return cs[0][0]
            stretches = sorted(stretches, key=by_first_day)

            # PLOT: variable thickness line
            for stretch, color in stretches:
                x, y = df.loc[stretch, [mob_col, Rt_col]].values.T
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                tests_per_hit = df.loc[stretch, 'tests_per_hit'].values
                np.place(tests_per_hit, np.isinf(tests_per_hit) | (tests_per_hit > 10000), 10000)
                z = 0.7*np.log(0 + tests_per_hit)
                np.place(z, np.isnan(z), 0)
                np.place(z, np.isinf(z), 1000)
                np.place(z, z < 0, 0)
                lwidths = [z]

                for segi, seg in enumerate(segments):
                    seg = seg.T
                    if kind == 'cases':  el = 0.15 + lwidths[0][segi] / 8
                    else:                el = 0.10 + lwidths[0][segi] / 14
                    co = sns.set_hls_values(color, l=el)
                    ax.plot(seg[0], seg[1], '-', color=co, linewidth=lwidths[0][segi],
                            alpha=1, solid_capstyle='round', zorder=20)

        elif kind == 'deaths': # ==---

            days_back = 14
            x, y = df[[mob_col, Rt_col]].values.T
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            de = df[['new_deaths14']]
            ca = df[['new_cases14' ]]
            ca = ca.set_index( ca.index.shift(+days_back, freq ='D') )  # <-- THIS
           #de = de.set_index( de.index.shift(-days_back, freq ='D') )  # <-- not this
            z = de.join(ca)
            z['cases14_per_death14'] = z['new_cases14'] / z['new_deaths14']
            z = z['cases14_per_death14'].values
            np.place(z, np.isnan(z), 0)
            np.place(z, np.isinf(z), 1000)
            np.place(z, z < 0, 0)
            lwidths = [1*np.log(1 + z)]

            for segi, seg in enumerate(segments):
                seg = seg.T
                if kind == 'cases':  el = 0.15 + lwidths[0][segi] / 8
                else:                el = 0.10 + lwidths[0][segi] / 14
                co = sns.set_hls_values(SNAIL_ORANGE, l=el)
                ax.plot(seg[0], seg[1], '-', color=co, linewidth=lwidths[0][segi],
                        alpha=1, solid_capstyle='round', zorder=20)

        # PLOT: dots + thin black
        x, y = df[[mob_col, Rt_col]].values.T
        ax.scatter(x, y, s=0.025, marker='o', facecolor='#000000', alpha=0.5, clip_on=True, zorder=30)
        ax.plot(x, y, linestyle='--', linewidth=0.1, color='#000000', alpha=0.33, zorder=40)

        # PLOT: panel title
        ax.annotate(text=country, xy=(0.5, 0.88), xycoords='axes fraction', fontsize=9, ha="center",
                   va="center", clip_on=False, zorder=100,
                    path_effects=[pthff.Stroke(linewidth=2, foreground=facecolor), pthff.Normal()])
        pop = population(country)
        total_per_1M = trajs_orig[country].loc[last_day, f"total_{kind}"] / pop
        heading = "{:d} {:s}/M".format(int(round(total_per_1M)), kind)
        ax.annotate(text=heading, xy=(0.5, 0.77), xycoords='axes fraction', fontsize=6.5,
                    ha="center", va="center", clip_on=False, zorder=100,
                    path_effects=[pthff.Stroke(linewidth=1.33, foreground=facecolor),
                                  pthff.Normal()])

        adjust_spines(ax, ['left', 'bottom'] if ax.is_first_col() else ['bottom'])
        set_ticks_lengths(ax)

    # PLOT: legend
    for ax in fig.axes:
        if ax.is_last_row() and ax.is_last_col():
            ax.set_axis_off()
    if kind == 'cases':
        put_legend_cases(fig.axes[-1], thr_weekly_cases_per_1M)
    elif kind == 'deaths':
        put_legend_deaths(fig.axes[-1])

    # PLOT: export and return
    fig.tight_layout(w_pad=0.4, h_pad=0.15)

    l, b, w, h = fig.axes[-1].get_position().bounds
    fig.axes[-1].set_position([l, b - 0.0185, w, h])

    fig.axes[-1].annotate('Last day:' + f" {final_day.strftime('%B %d, %Y')}",
                          xy=(0.0, 1.01), xycoords='axes fraction', color=ANNOT_COLOR)

    fn = f"Figure{fig_name}_{last_day.strftime('%b%d')}.pdf"
    fig.savefig(fn)
    print(f"Saved figure file {fn}.")
    return fig



def plot_cumulative_immobilization_and_gdp_drop(trajectories, locations, final_day, gdp_2020h1,
        fig_name):

    df = pd.DataFrame(columns='location cumul_2020H1_mobility_reduction gdp_2020H1_drop'.split())
    df = df.set_index('location')
    for loc in locations:
        if not loc in gdp_2020h1:
            print(f"{loc}: missing GDP data in figure {fig_name}")
            continue
        gdp_drop = -gdp_2020h1[loc]
        immob, _ = extract_cumulative_immobilization_and_deaths(trajectories, loc, 'daily').loc[final_day]
        df.loc[loc] = [immob, gdp_drop]

    fig, ax = plt.subplots(figsize=(5, 5))
    adjust_spines(ax, ['left', 'bottom'], left_shift=10)
    set_ticks_lengths(ax)
    ax.set_xlabel(r'Cumulated mobility reduction in the 1\textsuperscript{st} half of 2020')
    ax.set_ylabel(r'GDP loss in the 1\textsuperscript{st} half of 2020 (year-on-year \%)')
    ax.set_xlim((0, 5000))
    ax.set_ylim((-2, 14))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(*df.values.T)
    ax.plot([0, 5000], [intercept, intercept + slope*5000],
            linewidth=0.75, linestyle='--', color='#aaaaaa', zorder=5)

    weights = []
    for _, row in df.iterrows():
        location = row.name
        color = color_of(location)
        mob_red, gdp_drop = row[['cumul_2020H1_mobility_reduction', 'gdp_2020H1_drop']]
        ax.scatter([mob_red], [gdp_drop], color=color, zorder=10)
        ax.annotate(text=location.replace('United Kingdom', 'UK'),
                    xy=(mob_red + 49, gdp_drop + 0.028), xycoords='data',
                    color=sns.set_hls_values(color, l=0.3), fontsize=7, zorder=10)
        weights.append(population(location))

    rho, wrho = correlations(df.values, weights)
    ax.annotate(r'Correlation:',
                xy=(0.0, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
    ax.annotate(r"(non-weighted) Pearson's $\rho$ = " + f"{rho:.2f}",
                xy=(0.15, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
    ax.annotate(r"population-weighted Pearson's $\rho$ = " + f"{wrho:.2f}",
                xy=(0.15, 0.94), xycoords='axes fraction', color=ANNOT_COLOR)

    # export coordinates
    csv_fn = f"Figure{fig_name}.csv"
    np.savetxt(csv_fn, df.values, header='lockdown,gdp_loss', delimiter=',')

    # export image as PDF
    fig.tight_layout()
    fn = f"Figure{fig_name}.pdf"
    fig.savefig(fn)
    print(f"Saved figure file {fn}.")

    return fig



def plot_gdp_drop_and_excess_deaths(trajectories, locations, final_day, excess_deaths, gdp_2020h1,
        fig_name, scale_deaths=np.sqrt):

    fig, ax = plt.subplots(figsize=(5, 5))

    adjust_spines(ax, ['left', 'bottom'], left_shift=10)
    ax.set_xlabel(r'GDP loss in the 1\textsuperscript{st} half of 2020 (year-on-year \%)')
    ax.set_ylabel(r'$\sqrt{\textrm{\sf COVID-19-related deaths in the 1\textsuperscript{st} half of 2020 / M}}$')
    ax.set_xlim((-2, 14))
    make_sqrt_deaths_yaxis(ax)

    ed_locations = excess_deaths.keys()
    points, weights = [], []
    points_eur, weights_eur = [], []
    for loc in locations:
        if population(loc) < MIN_POPULATION_M or loc=='Serbia':
            print(f"{loc} skipped in figure {fig_name}")
            continue
        if loc not in ed_locations:
            print(f"{loc} in figure {fig_name}: deaths will be used in place of excess deaths")
        if loc not in gdp_2020h1:
            print(f"{loc} skipped in figure {fig_name} because of missing GDP data")
            continue

        is_in_Europe = not loc in STATE_TO_ABBREV and not loc in ['Canada', 'Taiwan', 'Japan', 'South Korea']
        deaths = max(excess_deaths[loc] if loc in excess_deaths else 0,
                     trajectories[loc].loc[final_day]['total_deaths'])
        x, y = -gdp_2020h1[loc], np.sqrt(deaths / population(loc) )
        put_final_dot(ax, loc, x, y, show_population_halo=True, label_shifting=False,
                      italic=not is_in_Europe)
        points.append([x, y])
        weights.append(population(loc))
        if is_in_Europe:
            points_eur.append([x, y])
            weights_eur.append(population(loc))

    values, values_eur = np.array(points), np.array(points_eur)
    rho, wrho = correlations(values, weights)
    rho_eur, wrho_eur = correlations(values_eur, weights_eur)
    ax.annotate(r'Correlation:',
                xy=(-0.01, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
    ax.annotate(r"(non-weighted) Pearson's $\rho$ = " + f"{rho:.2f} (Europe-only: {rho_eur:.2f})",
                xy=(0.155, 0.97), xycoords='axes fraction', color=ANNOT_COLOR)
    ax.annotate(r"population-weighted Pearson's $\rho$ = " + f"{wrho:.2f} (Europe-only: {wrho_eur:.2f})",
                xy=(0.155, 0.94), xycoords='axes fraction', color=ANNOT_COLOR)

    # export coordinates
    csv_fn = f"Figure{fig_name}_all.csv"
    np.savetxt(csv_fn, values, header='gdp_loss,sqrt_deaths', delimiter=',')
    csv_fn = f"Figure{fig_name}_eur.csv"
    np.savetxt(csv_fn, values_eur, header='gdp_loss,sqrt_deaths', delimiter=',')

    # export image as PDF
    fig.tight_layout()
    fn = f"Figure{fig_name}.pdf"
    fig.savefig(fn)
    print(f"Saved figure file {fn}.")

    return fig



if __name__ == '__main__':

    with gzip.open('processed_data.dill.gz', 'rb') as f:
        trajectories, locations, final_day, missing_days, excess_deaths, gdp_2020h1 = dill.load(f)

    print('Locations count:', len(locations))

    jul01 = pd.to_datetime('2020-07-01')

    fig1 = plot_cumulative_immobilization_and_deaths(trajectories, locations, [jul01, final_day],
            show_fronts=True, show_tail=False, show_corr_history=True, show_population_halo=True,
            fig_name='1')

    figS1 = plot_cumulative_immobilization_and_deaths(trajectories, locations, final_day,
             show_fronts=False, show_tail=True, show_corr_history=False, show_population_halo=False,
             fig_name='S1')

    fig2 = plot_R_vs_mobility_reduction(trajectories, locations, jul01, missing_days, fig_name='2')

    fig4 = plot_cumulative_immobilization_and_gdp_drop(trajectories, locations, jul01, gdp_2020h1,
            fig_name='4')

    fig5 = plot_gdp_drop_and_excess_deaths(trajectories, locations, jul01, excess_deaths,
            gdp_2020h1, fig_name='5')

