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

    # Change primary_gdp[Value_x] column name to primary_gdp[Primary_Energy_Supply]
    primary_gdp = primary_gdp.rename(columns={'Value_x': 'Primary_Energy_Supply'})
    primary_gdp = primary_gdp.rename(columns={'Value_y': 'GDP'})

    # The energy data (the Value column from the file primary_energy_supply.csv) are measured in million tons of oil.
    # Use the population table gdp.csv) and translate the energy data into million tons of oil per capita.
    primary_gdp['MLN_TOE_PER_CAP'] = primary_gdp['Primary_Energy_Supply'] / primary_gdp['GDP']

    # Create a new data frame for all the rows in primary_gdp where the LOCATION or TIME is missing
    missing_values = primary_gdp[primary_gdp.isnull().any(axis=1)]

    print("Missing values: ")
    # For each location in missing values, print the years for which the data is missing
    for location in missing_values['LOCATION'].unique():
        print(location, missing_values[missing_values['LOCATION'] == location]['TIME'].values)

    # Drop the rows where Value is missing
    primary_gdp = primary_gdp.dropna()

    # Drop Primary_Energy_Supply column
    primary_gdp = primary_gdp.drop(columns=['Primary_Energy_Supply'])

    return primary_gdp


def plot_multiple_locations(data, regions):
    '''
    This function takes a dataframe and a list of locations and plots the MLN_TOE_PER_CAP for each location in the list.
    :param df:
    :param location_list:
    :return: None
    '''
    # Calculate the number of plots and dimensions of the subplot grid
    num_plots = len(regions)
    num_rows = int(num_plots ** 0.5) + 1
    num_cols = int(num_plots ** 0.5) if num_plots % int(num_plots ** 0.5) == 0 else int(num_plots ** 0.5) + 1

    # Create the figure and subplots with shared x and y axes
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(10, 10))
    fig.suptitle('Energy Consumption vs GDP per Capita by Region')

    # Iterate over the regions and corresponding subplots
    for i, region in enumerate(regions):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_plots > 1 else axs

        # Filter the data for the current region
        region_data = data[data['LOCATION'] == region]
        years = region_data['TIME']
        energy = region_data['MLN_TOE_PER_CAP']
        gdp = region_data['GDP']

        # Plot the energy and GDP per capita data
        ax.plot(years, energy, label='Energy Consumption')
        ax.plot(years, gdp, label='GDP per Capita')
        ax.set_title(region)

        # Add legend to the subplot
        ax.legend()

    # Remove empty subplots
    if num_plots < num_rows * num_cols:
        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axs.flatten()[i])

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


df = prepare_data('primary_energy_supply.csv', 'world_population.csv', 'gdp.csv')
plot_multiple_locations(df, ['AUS', 'CAN', 'JPN', 'KOR', 'MEX', 'TUR', 'USA'])



