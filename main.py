from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# fetch dataset
gas_turbine_co_and_nox_emission_data_set = fetch_ucirepo(id=551)

# data (as pandas dataframes)
X = gas_turbine_co_and_nox_emission_data_set.data.features
y = gas_turbine_co_and_nox_emission_data_set.data.targets

# metadata
print(gas_turbine_co_and_nox_emission_data_set.metadata)
print("---------------------------------------------------------------------------------------------------------------")
# features table structure
print(X)
print("---------------------------------------------------------------------------------------------------------------")
