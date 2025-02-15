import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainigConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainigConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "SVR": SVR(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            # Get both the report and the tuned models
            model_results, tuned_models = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )
            for model_name, metrics in model_results.items():
                logging.info(
                    f"Model: {model_name}, Train R2: {metrics['train_score']:.4f}, Test R2: {metrics['test_score']:.4f}"
                )

            # Select the best model based on the test_score from the report
            best_model_name = max(
                model_results, key=lambda k: model_results[k]["test_score"]
            )
            best_model_score = model_results[best_model_name]["test_score"]
            best_model = tuned_models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} with Train R²: {model_results[best_model_name]['train_score']:.4f} and "
                f"Test R²: {best_model_score:.4f}"
            )

            if best_model_score < 0.75:
                raise CustomException("No good model found")
            logging.info("Best model found")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            raise CustomException(e, sys)
