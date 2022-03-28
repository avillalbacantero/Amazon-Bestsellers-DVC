"""This script contains all the data featurizing functionalities."""

import argparse
import logging
import os
import yaml

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def featurize(df: pd.DataFrame, num_components: int) -> np.ndarray:
    """Featurizes input data.

    Args:
        df (pd.DataFrame): The input DataFrame to featurize.

    Returns:
        pd.DataFrame: The featurized DataFrame.
    """

    # Apply TF-IDF feature extraction
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(df["Name"])
    logging.debug("Performed Feature Extraction using TD-IDF.")

    # Apply feature decomposition
    svd_truncer = TruncatedSVD(n_components=num_components, random_state=1)
    dec_mat = svd_truncer.fit_transform(sp_matrix)
    logging.debug("Performed Feature Decomposition using Truncated SVD.")

    return dec_mat


if __name__ == "__main__":

    # Configure a basic logger
    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process the dataset.")
    parser.add_argument("--input", type=str, help="Path/to the input CSV to featurize.")
    args = parser.parse_args()
    
    logging.info(f"Featurizing {args.input}")

    # Read the parameters to execute featurization
    params = yaml.safe_load(open("params.yaml"))["featurization"]

    # Read the dataset
    df = pd.read_csv(args.input)

    # DVC needs that the script creates the output directory
    OUTPUT_DIR = "./data/featurized"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    logging.info(f"Extracting features from: {args.input}")
    featurized_dataset = featurize(df, num_components=params["truncatedSVD_number_of_components"])
    logging.info(f"Features extracted: {args.input}")

    # Write intermediate result
    output_file = os.path.join(OUTPUT_DIR, "featurized_dataset")
    np.save(output_file, featurized_dataset)
    logging.info(f"Featurized dataset written to: {output_file}")
