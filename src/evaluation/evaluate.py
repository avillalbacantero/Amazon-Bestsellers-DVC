"""This script contains all the model evaluation functionalities."""

import argparse
from joblib import load
import json
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
    
    # Compute metrics and dump them to a JSON file
    mse = mean_squared_error(test_label, test_predictions)
    mae = mean_absolute_error(test_label, test_predictions) 
    
    logging.info(f"MSE (on test set) = {mse}")
    logging.info(f"MAE (on test set) = {mae}")
    
    scores = {"MSE": mse, "MAE": mae}
    with open("./reports/scores.json", 'w') as f:
        json.dump(scores, f, indent=4)
    
    # Plot predicted vs. actual and dump the values to a JSON file
    plt.figure()
    plt.title("Actual vs. Predicted Book Prices")
    sns.regplot(x=test_label, y=test_predictions)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    max_price = max(test_label) if max(test_label) >= max(test_predictions) else max(test_predictions)
    plt.xlim((0, max_price))
    plt.ylim((0, max_price))
    
    label_predictions = list()
    for i in range(len(test_label)):
        test_label_i = test_label[i]
        test_prediction_i = test_predictions[i]
        label_prediction = {"ground_truth": int(test_label_i), "predicted": int(test_prediction_i)}
        label_predictions.append(label_prediction)
        
    json_predictions = {"preds": label_predictions}
    with open("./reports/predictions.json", 'w') as f:
        json.dump(json_predictions, f, indent=4)
    
    # Plot Residuals    
    plt.figure()
    plt.title("Residual Plot")
    sns.residplot(x=test_predictions, y=test_label)
    plt.xlabel("Actual Price")
    plt.ylabel("Residual")
