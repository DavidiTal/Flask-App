import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import warnings
from car_data_prep import prepare_data

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the dataset
df = pd.read_csv('dataset.csv')

# Prepare the data
df_prepared = prepare_data(df)

# Identify original categorical columns before get_dummies
categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership']

# Split the data into features and target variable
X = df_prepared.drop(columns=['Price'])
y = df_prepared['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature names in the scaler
scaler.feature_names_in_ = X.columns

# Build the Elastic Net model
elastic_net = ElasticNet()

# Perform 10-fold cross-validation and print the average RMSE
cv_scores = cross_val_score(elastic_net, X_train_scaled, y_train, cv=10, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
average_rmse = cv_rmse.mean()

# Grid Search for best parameters
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100, 1000],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]
}
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters on the entire dataset
X_scaled = scaler.fit_transform(X)  # Refit scaler on entire data
scaler.feature_names_in_ = X.columns
elastic_net_best = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
elastic_net_best.fit(X_scaled, y)

# Save the model
model_file = open('trained_model.pkl', 'wb')
pickle.dump(elastic_net_best, model_file)
model_file.close()

# Save the scaler
scaler_file = open('scaler.pkl', 'wb')
pickle.dump(scaler, scaler_file)
scaler_file.close()
