import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import dagshub

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner='Pratik-1729', repo_name='MLOPS-EXPERIMENTS-WITH-MLFLOW', mlflow=True)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 120, 150, 170, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set experiment in MLflow
mlflow.set_experiment('breast_cancer-rf-ht')

# Enable autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Log best results manually
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    print("Best Params:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
