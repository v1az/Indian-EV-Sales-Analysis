import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('/kaggle/input/evsegm/IEA Global EV Data Clean 2024.csv')

# Check the first few rows and column data types
print(data.head())
print(data.info())

# Drop unnecessary columns
data = data[['mode', 'powertrain', 'year', 'value']]

# Convert columns to appropriate types
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['value'] = pd.to_numeric(data['value'], errors='coerce')

# Handle missing values if any
data.dropna(inplace=True)

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['mode', 'powertrain'])
print(data_encoded.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="whitegrid")

# Historical EV Sales by Mode
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='value', hue='mode', data=data, marker='o')
plt.title('Historical EV Sales by Mode')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles Sold')
plt.legend(title='Mode')
plt.grid(True)
plt.show()

# Historical EV Sales by Powertrain
plt.figure(figsize=(12, 6))
sns.lineplot(x='year', y='value', hue='powertrain', data=data, marker='o')
plt.title('Historical EV Sales by Powertrain')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles Sold')
plt.legend(title='Powertrain')
plt.grid(True)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Prepare data for clustering
features = data_encoded[['year', 'value']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Plot clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='year', y='value', hue='cluster', data=data, palette='Set1', s=100)
plt.title('Clustering of EV Sales')
plt.xlabel('Year')
plt.ylabel('Number of Vehicles Sold')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Prepare features and target
X = data_encoded.drop(columns=['value'])
y = data_encoded['value']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)

# Best parameters and model for Random Forest
print(f'Best Random Forest Parameters: {rf_grid_search.best_params_}')
best_rf_model = rf_grid_search.best_estimator_

# Hyperparameter tuning for Gradient Boosting (as an additional model)
gb_model = GradientBoostingRegressor(random_state=42)
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}
gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, n_jobs=-1, verbose=2)
gb_grid_search.fit(X_train, y_train)

# Best parameters and model for Gradient Boosting
print(f'Best Gradient Boosting Parameters: {gb_grid_search.best_params_}')
best_gb_model = gb_grid_search.best_estimator_

# Evaluate both models using cross-validation
rf_cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Random Forest Cross-Validated MAE: {-rf_cv_scores.mean()}')

gb_cv_scores = cross_val_score(best_gb_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f'Gradient Boosting Cross-Validated MAE: {-gb_cv_scores.mean()}')

# Choose the best model based on cross-validation MAE
if -rf_cv_scores.mean() < -gb_cv_scores.mean():
    print("Using Random Forest Model for Predictions")
    best_model = best_rf_model
else:
    print("Using Gradient Boosting Model for Predictions")
    best_model = best_gb_model

# Predict and evaluate on the test set
y_pred = best_model.predict(X_test)
print('Mean Absolute Error on Test Set:', mean_absolute_error(y_test, y_pred))

# Prepare future data for prediction
modes = data['mode'].unique()
powertrains = data['powertrain'].unique()
years = [2025]

# Create all combinations for future predictions
combinations = pd.DataFrame(
    [(year, mode, powertrain) for year in years for mode in modes for powertrain in powertrains],
    columns=['year', 'mode', 'powertrain']
)

# Encode future data
future_encoded = pd.get_dummies(combinations, columns=['mode', 'powertrain'])
future_encoded = future_encoded.reindex(columns=X.columns, fill_value=0)
future_encoded_scaled = scaler.transform(future_encoded)  # Apply scaling to future data

# Predict future sales using the best model
future_sales = best_model.predict(future_encoded_scaled)
combinations['predicted_sales'] = future_sales

# Show the results
print(combinations)

