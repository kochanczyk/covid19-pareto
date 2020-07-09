# COVID19&ndash;Pareto


Data &amp; code available in this repository feature the article "**Evaluation
of national responses to COVID-19 pandemic based on Pareto optimality**" by
Kochańczyk and Lipniacki (2020). The article is currently under consideration
in *Scientific Reports*. A preprint is available from
[medRxiv](https://doi.org/10.1101/2020.06.27.20141747).


### Data sources

All analyzed data originate from public sources:

* [Our World In Data](https://ourworldindata.org/coronavirus),
* [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility),
* [The COVID Tracking Project](https://covidtracking.com)

The master script, `analyze_data_and_make_figures.py`, may work either on
a static data snapshot, that is available from this repository, or on data
retrieved during script execution from online services. Script behavior is
controlled by the boolean value of `USE_DATA_SHAPSHOT` variable, which is 
currently set to True.

