"""This script contains all the data featurizing functionalities."""

import argparse
import logging
import os
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Configure a basic logger
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process the dataset.")
    parser.add_argument("--input", type=str, help="Path/to the input CSV to split.")
    parser.add_argument("--features", type=str, help="Path/to the features from the dataset.")
    args = parser.parse_args()
    
    # Read the parameters to execute featurization
    params = yaml.safe_load(open("params.yaml"))["split"]

    # Read the dataset
    df = pd.read_csv(args.input)
    features = np.load(args.features)

    # DVC needs that the script creates the output directory
    OUTPUT_DIR = "./data/processed"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    logging.info(f"Splitting {args.input}")
    test_percent = 1.0 - float(params["train_percentage"])
    train_x, test_x, train_y, test_y = train_test_split(features, df['Price'], test_size=test_percent, random_state=1)
    logging.info(f"Train and test sets generated from: {args.input}")

    # Write final result
    train_output_file = os.path.join(OUTPUT_DIR, "train_dataset")
    np.savez(train_output_file, train_x, train_y.values)
    logging.info(f"Saved {train_output_file}")
    
    test_output_file = os.path.join(OUTPUT_DIR, "test_dataset")
    np.savez(test_output_file, test_x, test_y.values)
    logging.info(f"Saved {test_output_file}")
