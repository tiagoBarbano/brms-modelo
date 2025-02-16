import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.create_experiment("experimento_a")    

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

with mlflow.start_run():
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(lr, "model")

