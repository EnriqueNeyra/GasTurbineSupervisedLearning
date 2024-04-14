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

# ---------------------------------------------------------------------------------------------------------------------#
# 1. need to split dataset into first 3 years (training) and the last 2 years (testing)
# 2. input training dataset into machine learning model
#       - Regression will predict continuous value (exact TEY value)
#       - Classification can predict category (TEY: low = 100 to 120, mid = 120 to 150, high = 150 to 180 )
# 3. input test dataset into model and evaluate performance

# TEY values histogram
plt.hist(X['TEY'], bins=30, color='blue', edgecolor='black')
plt.ylabel('Frequency')
plt.xlabel('TEY Value (MWH)')
plt.title('Histogram of TEY Values for All Data')
plt.show()

# splitting dataset
df = pd.DataFrame(X)
df_2011to2013 = df[df['year'].between(2011, 2013)] # training
df_2014to2015 = df[df['year'].between(2014, 2015)] # testing

# Step 1: Prepare the data (2011 to 2013 for training, and 2014 to 2015 for testing)
X_train = df_2011to2013.drop(columns=['TEY', 'year'])  # Features for training
X_test = df_2014to2015.drop(columns=['TEY', 'year'])  # Features for testing
y_train = df_2011to2013['TEY']  # Target variable for training (TEY)
y_test = df_2014to2015['TEY']  # Target variable for training (TEY)

# Step 3: Choose a regression model
model = LinearRegression()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Calculate residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.residplot(x=y_test, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 2})
plt.title('Residual Plot with Lowess Smoothing')
plt.xlabel('Actual TEY')
plt.ylabel('Residuals')
plt.axhline(y=0, color='blue', linestyle='--')  # Reference line at 0
plt.show()
