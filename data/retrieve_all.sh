#!/bin/bash

function download_datasets {

    # Our World In Data: cases & deaths
    wget https://covid.ourworldindata.org/data/owid-covid-data.csv

    # Our World In Data: tests
    wget https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv

    # Google COVID-19 Community Mobility Reports
    wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

    # Covid Tracking Project: cases, deaths, and tests in individual states of the USA
    wget https://covidtracking.com/api/v1/states/daily.csv
}


SNAPSHOT_FOLDER="snapshot-$(date -u +%Y%m%d)"

( mkdir ${SNAPSHOT_FOLDER} && cd ${SNAPSHOT_FOLDER} && download_datasets )

