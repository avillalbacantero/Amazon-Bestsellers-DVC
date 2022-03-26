import pandas as pd

from src.data.preprocess import preprocess


def test_preprocess(
    test_dataset: pd.DataFrame, preprocessed_test_dataset: pd.DataFrame
):
    """Checks if the preprocessing of the dataset is correct."""

    out_test_dataset = preprocess(test_dataset)
    assert out_test_dataset.equals(preprocessed_test_dataset)
