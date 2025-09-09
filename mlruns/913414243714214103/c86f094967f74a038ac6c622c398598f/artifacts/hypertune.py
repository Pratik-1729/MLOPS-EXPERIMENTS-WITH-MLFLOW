import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


data = load_breast_cancer()
X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target, name= 'target')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(random_state=42)

# determining the parameter grid for the gridsearchcv 
param_grid = {
    'n_estimators':[10,50,100,120,150,170,200],
    'max_depth':[None,10,20,30]
}

# applying gridsearchcv
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)

# grid_search.fit(X_train,y_train)

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(best_params)
# print(best_score)

# WE are using mlflow
mlflow.set_experiment('breast_cancer-rf-ht')
with mlflow.start_run():
    grid_search.fit(X_train,y_train)

    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # LOGGING PARAMS
    mlflow.log_params(best_params)

    # logging accuracy
    mlflow.log_metric("accuracy",best_score)

    # log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"training")

    # log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,"testing")

    # log source code
    mlflow.log_artifact(__file__)

    # log model
    mlflow.sklearn.log_model(grid_search.best_estimator_,'random_Forest')

    # set tags
    mlflow.set_tag("author","Pratik Gaikwad")

    print(best_params)
    print(best_score)


