"""This script contains all the model evaluation functionalities."""

import argparse
from joblib import load
import logging
import yaml

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == "__main__":

    # Configure a basic logger
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Evaluates a ML model for Price Prediction."
    )
    parser.add_argument("--test_data", type=str, help="Path/to the test NPZ data.")
    parser.add_argument("--model", type=str, help="Path/to the trained model to evaluate.")
    args = parser.parse_args()
    
    # Read the parameters to execute featurization
    params = yaml.safe_load(open("params.yaml"))["evaluation"]

    # Read the test data
    test_data = np.load(args.test_data)
    test_features = test_data["data_features"]
    test_label = test_data["label"]
    logging.info(f"Test data {args.test_data} successfully loaded")

    # Make the predictions with the already trained model
    trained_model = load(args.model)
    logging.info(f"Model {args.model} successfully loaded")
    test_predictions = trained_model.predict(test_features)
    logging.info(f"Test set predictions done. Evaluating the model...")
    
    # Compute metrics
    mse = mean_squared_error(test_label, test_predictions)
    mae = mean_absolute_error(test_label, test_predictions) 
    logging.info(f"MSE (on test set) = {mse}")
    logging.info(f"MAE (on test set) = {mae}")
    
    # Plot predicted vs. actual
    plt.figure(1)
    plt.title("Actual vs. Predicted Book Prices")
    sns.regplot(x=test_label, y=test_predictions)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    
    # Plot Residuals    
    plt.figure(2)
    plt.title("Residual Plot")
    sns.residplot(x=test_predictions, y=test_label)
    plt.xlabel("Actual Price")
    plt.ylabel("Residual")
