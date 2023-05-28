import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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

    # Remove rows where TIME is not in the range 1960-2017
    primary_gdp = primary_gdp[(primary_gdp['TIME'] >= 1960) & (primary_gdp['TIME'] <= 2017)]

    # Change location['Country Code'] to location['LOCATION']
    population = population.rename(columns={'Country Code': 'LOCATION'})

    # Merge the DataFrames based on country code and year
    merged_df = primary_gdp.merge(population, on='LOCATION')
    # Init a new column for GDP per capita
    merged_df['GDP per capita'] = 0

    # Divide GDP value by population where primary_gdp[LOCATION] == population[Country Code] and primary_gdp[TIME] == population[primary_gdp[TIME]]
    for index, row in merged_df.iterrows():
        time = merged_df.loc[index, 'TIME']
        # time to string
        time = str(time)
        merged_df.loc[index, 'GDP per capita'] = merged_df.loc[index, 'GDP'] / merged_df.loc[index, time]

    primary_gdp['GDP'] = merged_df['GDP per capita']

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
    :param data:
    :param regions:
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
        # ax.plot(years, energy, label='Energy Consumption')
        ax.plot(energy, gdp, label='GDP per Capita')
        ax.set_title(region)

        # Add legend to the subplot
        ax.legend()

    # Remove empty subplots
    if num_plots < num_rows * num_cols:
        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axs.flatten()[i])

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

    return fig


def plot_multiple_locations_regressions(data, regions):
    '''
    This function takes a dataframe and a list of locations and plots the MLN_TOE_PER_CAP for each location in the list,
    compared to GDP per capita.
    :param data:
    :param regions:
    :return:
    '''
    num_plots = len(regions)
    num_rows = int(num_plots ** 0.5) + 1
    num_cols = int(num_plots ** 0.5) if num_plots % int(num_plots ** 0.5) == 0 else int(num_plots ** 0.5) + 1

    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(10, 10))
    fig.suptitle('Energy per Capita vs GDP per Capita by Area')

    axs_flat = axs.flatten() if num_plots > 1 else [axs]

    for i, area in enumerate(regions):
        ax = axs_flat[i]

        area_data = data[data['LOCATION'] == area]
        energy = area_data['MLN_TOE_PER_CAP']
        gdp = area_data['GDP']

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(energy, gdp)

        # Generate points for regression line
        x_vals = np.linspace(energy.min(), energy.max(), 100)
        y_vals = slope * x_vals + intercept

        ax.plot(energy, gdp, 'o', label='Data')
        ax.plot(x_vals, y_vals, 'r-', label='Regression Line')
        ax.set_title(f'{area} - R^2: {r_value ** 2:.2f}')

        ax.legend()

    if num_plots < num_rows * num_cols:
        empty_subplots = axs_flat[num_plots:]
        for ax in empty_subplots:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

    return fig


def plot_multiple_years_regressions(data, years, remove_outliers=False):
    num_plots = len(years)
    num_rows = int(num_plots ** 0.5) + 1
    num_cols = int(num_plots ** 0.5) if num_plots % int(num_plots ** 0.5) == 0 else int(num_plots ** 0.5) + 1

    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(10, 10))
    fig.suptitle('Energy per Capita vs GDP per Capita by Year')

    years_sorted = sorted(years)

    for i, year in enumerate(years_sorted):
        if num_plots > 1:
            ax = axs[i // num_cols, i % num_cols]
        else:
            ax = axs

        year_data = data[data['TIME'] == year]
        locations = year_data['LOCATION']
        energy = year_data['MLN_TOE_PER_CAP']
        gdp = year_data['GDP']

        if remove_outliers:
            z_scores = stats.zscore(gdp)
            outliers = np.abs(z_scores) > 3
            locations = locations[~outliers]
            energy = energy[~outliers]
            gdp = gdp[~outliers]
            removed_regions = set(year_data['LOCATION'].unique()) - set(locations.unique())
            if removed_regions:
                print(f"Removed outliers for year {year} in the following regions: {', '.join(removed_regions)}")

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(energy, gdp)

        # Generate points for regression line
        x_vals = np.linspace(energy.min(), energy.max(), 100)
        y_vals = slope * x_vals + intercept

        ax.plot(energy, gdp, 'o', label='Energy Per Capita VS GDP')
        ax.plot(x_vals, y_vals, 'r-', label='Regression Line')
        ax.set_title(f'Year: {year} - R^2: {r_value ** 2:.2f}')

        ax.legend()

    if num_plots < num_rows * num_cols:
        empty_subplots = axs.flatten()[num_plots:]
        for ax in empty_subplots:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

    return fig


df = prepare_data('primary_energy_supply.csv', 'world_population.csv', 'gdp.csv')
print(df.head())
# years = [2010, 2015, 2020, 1998, 2000, 2007]
# figure = plot_multiple_locations(df, ['USA', 'CHN', 'IND', 'RUS', 'JPN', 'DEU', 'GBR', 'FRA', 'ITA', 'BRA', 'CAN', 'AUS'])
# figure.show()
