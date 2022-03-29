"""This script contains all the model training functionalities."""

import argparse
from joblib import dump
import logging
import os
import yaml

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    # Configure a basic logger
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Trains a ML model for Price Prediction."
    )
    parser.add_argument("--train_data", type=str, help="Path/to the train NPZ data.")
    args = parser.parse_args()
    
    # Read the parameters to execute featurization
    params = yaml.safe_load(open("params.yaml"))["training"]

    # Read the training data
    train_data = np.load(args.train_data)
    train_features = train_data["data_features"]
    train_label = train_data["label"]
    logging.info(f"Training data {args.train_data} successfully loaded")

    # DVC needs that the script creates the output directory
    OUTPUT_DIR = "./models"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Create the model
    linear_regression_pipeline = Pipeline(steps=[("model", LinearRegression())])

    # Train the Linear Regresion
    logging.info("Training a Linear Regression model...")
    linear_regression_pipeline.fit(train_features, train_label)

    # Write the trained model
    output_file = os.path.join(OUTPUT_DIR, "linear_regression.joblib")
    dump(linear_regression_pipeline, output_file)
    logging.info(f"Trained model written to: {output_file}")
