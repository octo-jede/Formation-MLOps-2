import os
import time

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_model_with_io(features_path: str, model_registry_folder: str) -> None:
    features = pd.read_parquet(features_path)

    train_model(features, model_registry_folder)


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    with mlflow.start_run():
        mlflow.sklearn.autolog(log_models=True)
        model = RandomForestRegressor(n_estimators=1, max_depth=10, n_jobs=1)
        model.fit(X, y)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    joblib.dump(model, os.path.join(model_registry_folder, time_str + '.joblib'))


def predict_with_io(features_path: str, model_path: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)
    features = predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features


def new_train_model_with_io(features_path: str, model_registry_folder: str) -> None:
    features = pd.read_parquet(features_path)

    new_train_model(features, model_registry_folder)


def new_train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    with mlflow.start_run():
        mlflow.xgboost.autolog(log_models=True)  # mlflow.sklearn.autolog(log_models=True)
        model = XGBRegressor(n_estimators=10, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, eval_metric="rmse")  # XGBRegressor(eta=.1, max_depth=20, min_child_weight=3, subsample=.8, sampling_method="gradient_based")
        # model = RandomForestRegressor(n_estimators=10, max_depth=20, n_jobs=1)  #min_samples_split=6, min_samples_leaf=3,
        model.fit(X, y)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    joblib.dump(model, os.path.join(model_registry_folder, time_str + '.joblib'))


def new_predict_with_io(features_path: str, model_path: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)
    features = new_predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def new_predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features['predictions'] = model.predict(features)
    return features