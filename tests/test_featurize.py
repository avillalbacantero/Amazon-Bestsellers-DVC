import pandas as pd
import numpy as np

from src.features.featurize import featurize


def test_featurize(
    preprocessed_test_dataset: pd.DataFrame, featurized_test_dataset: np.ndarray
):
    """Checks if the preprocessing of the dataset is correct."""

    out_test_dataset = featurize(preprocessed_test_dataset, num_components=15)
    np.testing.assert_equal(out_test_dataset, featurized_test_dataset)
