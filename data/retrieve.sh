#!/bin/bash

# Automatically retrieved
# -----------------------

function download_datasets {

    ## Our World In Data: cases & deaths
    wget https://covid.ourworldindata.org/data/owid-covid-data.csv

    ## Our World In Data: tests
    wget https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv

    ## Google COVID-19 Community Mobility Reports
    wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
    #
    # Note: as of July 6, data for Serbia are missing, so the following snapshot has been retrieved instead:
    # https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/6674288a0cef84676a09edbcc46efd6659896e5f/google_reports/Global_Mobility_Report.csv

    ## Covid Tracking Project: cases, deaths, and tests in individual states of the USA
    wget https://covidtracking.com/api/v1/states/daily.csv
}


SNAPSHOT_FOLDER="snapshot-$(date -u +%Y%m%d)"

( mkdir ${SNAPSHOT_FOLDER} && cd ${SNAPSHOT_FOLDER} && download_datasets && echo "Files stored in ${SNAPSHOT_FOLDER}.")



# Manually retrieved
# ------------------

echo "Data from Eurostat and CDC have are to be manually retrieved from these websites:"

# demo_r_mwk_ts_1_Data.csv.bz2
echo "* https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Weekly_death_statistics"
# see also: https://ec.europa.eu/eurostat/statistics-explained/images/d/da/Weekly_deaths_15_10_2020-update.xlsx

# Excess_Deaths_Associated_with_COVID-19.csv
echo "* https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm"

# estat_namq_10_gdp--SCA.csv.bz2
echo "* https://ec.europa.eu/eurostat/databrowser/view/namq_10_gdp/default/table?lang=en"
