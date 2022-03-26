import pandas as pd
import pytest


@pytest.fixture
def test_dataset() -> pd.DataFrame:
    """A pd.DataFrame of a Amazon-Best-Sellers-Books test dataset."""
    return pd.read_csv("./test_data/test_dataset.csv")


@pytest.fixture
def preprocessed_test_dataset() -> pd.DataFrame:
    """A pd.DataFrame with the Amazon-Best-Sellers-Books preprocessed
    test dataset.
    """
    return pd.read_csv("./test_data/ground_truth_preprocessed_test_dataset.csv")
