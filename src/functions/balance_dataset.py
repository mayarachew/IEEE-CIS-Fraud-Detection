"""Function to balance dataset using over sampling"""
from imblearn.over_sampling import RandomOverSampler  # type: ignore


def balance_dataset(X, y, proportion):
    oversample = RandomOverSampler(sampling_strategy=proportion)
    X, y = oversample.fit_resample(X, y)

    return X, y