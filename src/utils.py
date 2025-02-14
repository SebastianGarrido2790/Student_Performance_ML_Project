import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


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


def evaluate_models(X_train, y_train, X_test, y_test, models, cv=5):
    try:
        report = {}
        for model_name, model in models.items():
            # Perform cross validation on the training data
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

            # Fit the model on the full training set
            model.fit(X_train, y_train)

            # Predict on the test set
            y_test_pred = model.predict(X_test)

            # Calculate the R² score on the test set
            test_score = r2_score(y_test, y_test_pred)

            # Store the evaluation metrics in the report
            report[model_name] = {
                "cv_mean": np.mean(cv_scores),
                "cv_std": np.std(cv_scores),
                "test_score": test_score,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
