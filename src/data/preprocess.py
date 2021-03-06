"""This script contains all the data preprocessing functionalities."""

import argparse
import logging
import string
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    df["Name"] = df["Name"].apply(lambda x: x.lower())
    df["Name"] = df["Name"].apply(
        lambda x: x.translate(str.maketrans(" ", " ", string.punctuation))
    )
    logging.debug("Cleaned `Name` column.")

    nltk.download("vader_lexicon")  # downloads lexicon if not exists
    sid = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Name"].apply(lambda x: sid.polarity_scores(x))
    df["Positive Sentiment"] = df.Sentiment.apply(lambda x: x["pos"])
    df["Neutral Sentiment"] = df.Sentiment.apply(lambda x: x["neu"])
    df["Negative Sentiment"] = df.Sentiment.apply(lambda x: x["neg"])
    df["Compound Sentiment"] = df.Sentiment.apply(lambda x: x["compound"])
    df = df.drop(columns=["Sentiment"])
    logging.debug(
        "Sentiment and Compound sentiment have been calculated over the `Name` column."
    )

    return df


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Process the dataset.")
    parser.add_argument("--input", type=str, help="Path/to the input CSV to process.")
    args = parser.parse_args()

    logging.info(f"Preprocessing {args.input}")
    df = pd.read_csv(args.input)

    # DVC needs that the script creates the output directory
    OUTPUT_DIR = "./data/preprocessed"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    logging.info(f"Preprocessing: {args.input}")
    preprocessed_df = preprocess(df)
    logging.info(f"Preprocessed: {args.input}")

    # Write intermediate result
    output_file = os.path.join(OUTPUT_DIR, "preprocessed_dataset.csv")
    preprocessed_df.to_csv(output_file, index=False)
    logging.info(f"Preprocessed dataset written to: {output_file}")
