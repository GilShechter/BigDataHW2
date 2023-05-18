import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(primary_path, population_path, gdp_path):
    '''
    This function prepares the data for the analysis. It takes the paths to 3 files and loads them into pandas dataframes.
    The function will return a new Dataframe containing the data from all 3 files: Location, time, GDP and Primary energy supply.
    :param primary_path:
    :param population_path:
    :param gdp_path:
    :return: Dataframe
    '''
    # Load the data
    primary = pd.read_csv(primary_path, header=0)
    population = pd.read_csv(population_path, header=0)
    gdp = pd.read_csv(gdp_path, header=0)

    # In primary, keep only the rows where MEASURE is in MLN_TOE
    primary = primary[primary['MEASURE'] == 'MLN_TOE']

    # In gdp, keep only the rows where MEASURE is in USD_CAP
    gdp = gdp[gdp['MEASURE'] == 'USD_CAP']

    # In primary and gdp, keep only the LOCATION, TIME and Value columns
    primary = primary[['LOCATION', 'TIME', 'Value']]
    gdp = gdp[['LOCATION', 'TIME', 'Value']]

    # Use pandas merge to keep only values that are in both primary and gdp
    primary_gdp = pd.merge(primary, gdp, on=['LOCATION', 'TIME'])

    # The energy data (the Value column from the file primary_energy_supply.csv) are measured in million tons of oil.
    # Use the population table gdp.csv) and translate the energy data into million tons of oil per capita.
    primary_gdp['Value_Per_Cap'] = primary['Value'] / gdp['Value']

    # Create a new data frame for missing values
    missing_values = primary_gdp[primary_gdp.isnull().any(axis=1)]

    # For each location in missing values, print the years for which the data is missing
    for location in missing_values['LOCATION'].unique():
        print(location, missing_values[missing_values['LOCATION'] == location]['TIME'].values)

    # Drop the rows where Value is missing
    primary_gdp = primary_gdp.dropna()



prepare_data('primary_energy_supply.csv', 'world_population.csv', 'gdp.csv')



