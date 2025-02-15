import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import yaml


def save_object(file_path, obj):
    """
    This function is responsible for saving the object to the file
    param file_path: path where the object needs to be saved
    param obj: object to be saved
    return: None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_param_grids(yaml_path: str) -> dict:
    """
    Loads hyperparameter grids from a YAML file located relative to this file.
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_yaml_path = os.path.join(base_path, yaml_path)
        with open(full_yaml_path, "r") as file:
            param_grids = yaml.safe_load(file)
        return param_grids
    except Exception as e:
        raise Exception(f"Failed to load YAML file: {e}")


def evaluate_models(
    X_train, y_train, X_test, y_test, models, cv=5, yaml_path="param_grids.yaml"
):
    """
    Evaluates each model using hyperparameter tuning with GridSearchCV (if a grid is provided),
    cross validation, and test set performance. Returns both a performance report and a dictionary
    of tuned (and fitted) model instances.
    """
    try:
        report = {}
        tuned_models = {}

        # Load hyperparameter grids from YAML file
        param_grids = load_param_grids(yaml_path)

        for model_name, model in models.items():
            # Retrieve hyperparameter grid for the current model
            param_grid = param_grids.get(model_name, {})

            # Perform hyperparameter tuning if a non-empty grid is provided
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=cv,
                    scoring="r2",
                    n_jobs=-1,
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                best_model = model
                best_params = {}

            # Evaluate the tuned model using cross validation
            cv_scores = cross_val_score(
                best_model, X_train, y_train, cv=cv, scoring="r2"
            )

            # Fit the best model on the full training data
            best_model.fit(X_train, y_train)

            # Predict on the test set and calculate the R² score
            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            train_score = r2_score(y_train, best_model.predict(X_train))

            # Record the results
            report[model_name] = {
                "train_score": train_score,
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores),
                "test_score": test_score,
                "best_params": best_params,
            }
            # Save the tuned (and fitted) model
            tuned_models[model_name] = best_model

        return report, tuned_models

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    This function is responsible for loading the object from the file
    param file_path: path where the object needs to be loaded
    return: object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
