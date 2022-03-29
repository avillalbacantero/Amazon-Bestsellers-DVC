import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_dataset() -> pd.DataFrame:
    """A pd.DataFrame of a Amazon-Best-Sellers-Books test dataset."""
    return pd.read_csv("./tests/test_data/test_dataset.csv")


@pytest.fixture
def preprocessed_test_dataset() -> pd.DataFrame:
    """A pd.DataFrame with the Amazon-Best-Sellers-Books preprocessed
    test dataset.
    """
    return pd.read_csv("./tests/test_data/preprocessed_test_dataset.csv")


@pytest.fixture
def featurized_test_dataset() -> np.ndarray:
    """A np.ndarray with the features extracted from the 
    Amazon-Best-Sellers-Books test dataset. 15 components for feature
    decomposition with Truncated SVD were used.

    Returns:
        np.ndarray: The array with the extracted features.
    """
    
    return np.load("./tests/test_data/featurized_test_dataset.npz")
