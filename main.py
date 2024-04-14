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
# features table structure (TEY is the target variable)
print(X)
print("---------------------------------------------------------------------------------------------------------------")

# 1. need to split dataset into first 3 years (training) and the last 2 years (testing)
# 2. input training dataset into machine learning model
#       - Regression will predict continuous value (exact TEY value)
#       - Classification can predict category (different ranges of TEY values such as low, mid, high)
# 3. input test dataset into model and evaluate performance
