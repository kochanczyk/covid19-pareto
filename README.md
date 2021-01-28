# COVID19&ndash;Pareto


Data &amp; code available in this repository feature the article "Pareto-based evaluation
of national responses to COVID-19 pandemic shows that saving lives and protecting economy
are non-trade-off objectives" by Kochańczyk &amp; Lipniacki (2021) published in *Scientific Reports* **11**:2425
(DOI: [10.1038/s41598-021-81869-2](https://dx.doi.org/10.1038/s41598-021-81869-2)).


### Data sources

All analyzed data originate from public sources:

* [Our World In Data](https://ourworldindata.org/coronavirus),
* [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility),
* [The COVID Tracking Project](https://covidtracking.com)
* [Eurostat](https://ec.europa.eu/eurostat)
* [Center for Disease Control](https://www.cdc.gov)


### Source code

The scripts are intended to be executed in sequential order:

* first `01-process_data.py`,
* then `02-make_figures.py`.
