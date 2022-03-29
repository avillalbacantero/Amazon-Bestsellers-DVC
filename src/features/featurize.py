"""This script contains all the data featurizing functionalities."""

import argparse
import logging
import os
from sklearn.preprocessing import LabelEncoder
import yaml

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def encode_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes the `Genre` columns using values between 0 and
    `num_classes -1`.

    Args:
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: Encoded DataFrame
    """
    label_encoder = LabelEncoder()
    df["Genre"] = label_encoder.fit_transform(df["Genre"])
    return df


def featurize_name(name_series: pd.Series, num_components: int) -> np.ndarray:
    """Featurizes the `Name` of the books data.

    Args:
        df (pd.Series): The Series of `Name` to featurize.

    Returns:
        np.ndarray: The name features
    """

    # Apply TF-IDF feature extraction to the `Name` column
    vectorizer = TfidfVectorizer()
    sp_matrix = vectorizer.fit_transform(name_series)
    logging.debug("Performed Feature Extraction using TD-IDF.")

    # Apply feature decomposition to those features
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

    # Feature extraction
    logging.info(f"Extracting name features from: {args.input}")
    featurized_name = featurize_name(
        df["Name"], num_components=params["truncatedSVD_number_of_components"]
    )
    logging.info(f"Name features extracted: {args.input}")

    logging.info(f"Encoding data from: {args.input}")
    df_label = df["Price"]
    df = df.drop(columns=["Price"])
    encoded_df = encode_genre(df)
    logging.info(f"Data has been encoded: {args.input}")

    # Drop the columns the solution does not use
    encoded_df = encoded_df.drop(
        columns=[
            "Name",
            "Author",
            "User Rating",
            "Reviews",
            "Positive Sentiment",
            "Neutral Sentiment",
            "Negative Sentiment",
        ]
    )

    # Concatenate the features
    data_features = encoded_df.to_numpy()
    data_features = np.hstack((featurized_name, data_features))

    # Write intermediate result
    output_file = os.path.join(OUTPUT_DIR, "featurized_dataset")
    np.savez_compressed(output_file, data_features=data_features, label=df_label.values)
    logging.info(f"Featurized dataset written to: {output_file}")
