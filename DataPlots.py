import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 150)


# fetch dataset
gas_turbine_co_and_nox_emission_data_set = fetch_ucirepo(id=551)

# data (as pandas dataframes)
X = gas_turbine_co_and_nox_emission_data_set.data.features
y = gas_turbine_co_and_nox_emission_data_set.data.targets

# metadata
print(gas_turbine_co_and_nox_emission_data_set.metadata)
print("---------------------------------------------------------------------------------------------------------------")
# features table structure (TEY is the target variable)
print(X)
print("---------------------------------------------------------------------------------------------------------------")

df = pd.DataFrame(X)

# PLOT OF TIT vs TAT with TEY
# Create the scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['TIT'], df['TAT'], c=df['TEY'], cmap='viridis')
plt.colorbar(scatter, label='Turbine Energy Yield (TEY)')

# Adding labels and title
plt.xlabel('Turbine Inlet Temperature (TIT)')
plt.ylabel('Turbine After Temperature (TAT)')
plt.title('Scatter Plot of TIT vs TAT Colored by TEY')

# Show the plot
plt.show()


# PLOT OF AFDP vs CDP with TEY
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['AFDP'], df['CDP'], c=df['TEY'], cmap='viridis')
plt.colorbar(scatter, label='Turbine Energy Yield (TEY)')

# Adding labels and title
plt.xlabel('Air Filter Difference Pressure (AFDP)')
plt.ylabel('Compressor Discharge Pressure (CDP)')
plt.title('Scatter Plot of AFDP vs CDP Colored by TEY')

# Show the plot
plt.show()


# PLOT OF GTEP vs CDP with TEY
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['GTEP'], df['CDP'], c=df['TEY'], cmap='viridis')
plt.colorbar(scatter, label='Turbine Energy Yield (TEY)')

# Adding labels and title
plt.xlabel('Gas Turbine Exhaust Pressure (GTEP)')
plt.ylabel('Compressor Discharge Pressure (CDP)')
plt.title('Scatter Plot of GTEP vs CDP Colored by TEY')

# Show the plot
plt.show()