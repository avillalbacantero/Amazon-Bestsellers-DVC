"""This script contains all the data featurizing functionalities."""

import argparse
import logging
import os
import yaml

import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Configure a basic logger
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process the dataset.")
    parser.add_argument(
        "--features", type=str, help="Path/to the features from the dataset."
    )
    args = parser.parse_args()

    # Read the parameters to execute featurization
    params = yaml.safe_load(open("params.yaml"))["split"]

    # Read the dataset
    all_features_label = np.load(args.features)
    features = all_features_label["data_features"]
    label = all_features_label["label"]
    
    # DVC needs that the script creates the output directory
    OUTPUT_DIR = "./data/processed"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    logging.info(f"Splitting {args.features} into train and test...")
    test_percent = 1.0 - float(params["train_percentage"])
    train_features, test_features, train_label, test_label = train_test_split(
        features, label, test_size=test_percent, random_state=1
    )
    logging.info(f"Train and test sets generated from: {args.features}")

    # Write final result
    train_output_file = os.path.join(OUTPUT_DIR, "train_dataset")
    np.savez_compressed(
        train_output_file,
        data_features=train_features,
        label=train_label
    )
    logging.info(f"Saved {train_output_file}")

    test_output_file = os.path.join(OUTPUT_DIR, "test_dataset")
    np.savez_compressed(
        test_output_file,
        data_features=test_features,
        label=test_label
    )
    logging.info(f"Saved {test_output_file}")
