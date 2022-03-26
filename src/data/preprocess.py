"""This script contains all the data preprocessing functionalities."""

import argparse
import logging
import string

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

    nltk.download("vader_lexicon")  # downloads lexicon if not exists
    sid = SentimentIntensityAnalyzer()
    df["Sentiment"] = df["Name"].apply(lambda x: sid.polarity_scores(x))
    df["Positive Sentiment"] = df.Sentiment.apply(lambda x: x["pos"])
    df["Neutral Sentiment"] = df.Sentiment.apply(lambda x: x["neu"])
    df["Negative Sentiment"] = df.Sentiment.apply(lambda x: x["neg"])
    df["Compound Sentiment"] = df.Sentiment.apply(lambda x: x["compound"])
    df = df.drop(columns=["Sentiment"])

    return df


if __name__ == "__main__":

    log_fmt = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description="Process the dataset.")
    parser.add_argument("--input", type=str, help="Path/to the input CSV to process.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    logging.info(f"Preprocessing: {args.input}")
    preprocessed_df = preprocess(df)
    logging.info(f"Preprocessed: {args.input}")
