# Script Objective: Read the data from the source and store it in the database

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainigConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        """
        Reads data from the source, splits it into train and test sets,
        and saves the sets to specified paths.
        """
        logging.info("Starting Data Ingestion")
        try:
            # Reading the data
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Data successfully read from source")

            # Ensuring directory structure is in place
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save raw data (optional, useful for debugging or archiving)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            # Splitting data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets")

            # Saving train and test data
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")

            logging.info("Data Ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("An error occurred during Data Ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data, test_data
        )
        logging.info("Data Transformation completed successfully")

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        best_model_name, best_r2_score = model_trainer.initiate_model_trainer(
            train_arr, test_arr
        )

        # Log final model performance
        logging.info("Pipeline execution completed successfully")
        print(f"Best Model: {best_model_name} with R² score: {best_r2_score}")

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {str(e)}")
        raise CustomException(e, sys)
