import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd

def train_model(X_train, y_train, model_type="Random Forest", param_grid=None):
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        default_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
    elif model_type == "XGBoost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        default_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        default_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    elif model_type == "Linear Regression":
        model = LinearRegression()
        default_param_grid = {}  # Regressão linear não possui hiperparâmetros padrão
    else:
        raise ValueError(f"Modelo {model_type} não suportado.")

    if param_grid is None:
        param_grid = default_param_grid

    if param_grid:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        model.fit(X_train, y_train)
        best_model = model
        best_params = {}

    return best_model, best_params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Validação cruzada para garantir avaliação robusta
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    mean_cv_mse = -cv_scores.mean()

    return mae, mse, r2, mean_cv_mse


def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def analyze_feature_importance(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
            'Variavel': X_train.columns,
            'Importancia': importances
        }).sort_values(by='Importancia', ascending=False).dropna()
    return importance_df