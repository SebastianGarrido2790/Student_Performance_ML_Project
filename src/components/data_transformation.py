import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocess.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        logging.info("Data Transformation started")
        try:
            numerical_columns = ["math_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function starts the data transformation process
        """
        logging.info("Initiate the Data Transformation Process")
        try:
            train_path = pd.read_csv(train_path)
            test_path = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")
            target_column = "writing_score"
            numerical_columns = ["math_score", "reading_score"]

            input_feature_train_df = train_path.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_path[target_column]
            input_feature_test_df = test_path.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_path[target_column]
            logging.info(
                "Train and test data are separated into input and target features"
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Save the preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
