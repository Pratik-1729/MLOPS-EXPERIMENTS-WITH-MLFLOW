import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("wine_random_forest")
wine = load_wine()
X = wine.data
y = wine.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)

max_depth = 10
n_estimators = 10



with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth,random_state=42,n_estimators=n_estimators)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    # confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.title('Confusion matrix')

    plt.savefig('confusion_matrix.png')
    # log artifacts using mlflow
    mlflow.log_artifact('confusion_matrix.png')
    if "__file__" in globals() and os.path.exists(__file__):
        mlflow.log_artifact(__file__)

    mlflow.set_tags({'Author':'Pratik','Project':'Wine classification'})

    mlflow.sklearn.log_model(rf,'Random forest model')

    print(accuracy)