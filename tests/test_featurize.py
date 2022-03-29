import pandas as pd
import numpy as np

from src.features.featurize import featurize_name


def test_featurize_name(
    preprocessed_test_dataset: pd.DataFrame, featurized_test_dataset: np.ndarray
):
    """Checks if the preprocessing of the dataset is correct."""

    NUM_COMPONENTS = 2
    out_test_dataset = featurize_name(preprocessed_test_dataset["Name"], NUM_COMPONENTS)
    ground_truth_name_features = featurized_test_dataset["data_features"][:, 0:NUM_COMPONENTS]
    np.testing.assert_equal(out_test_dataset, ground_truth_name_features)
