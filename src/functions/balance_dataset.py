"""Function to balance dataset using over sampling"""
from imblearn.over_sampling import RandomOverSampler  # type: ignore
from imblearn.under_sampling import RandomUnderSampler  # type: ignore

SEED_VAL = 42

def balance_dataset(technique, X, y, proportion):
    if technique == 'oversampling':
        sampling = RandomOverSampler(sampling_strategy=proportion, random_state=SEED_VAL)    
    elif technique == 'undersampling':
        sampling = RandomUnderSampler(sampling_strategy=proportion, random_state=SEED_VAL)
        
    X, y = sampling.fit_resample(X, y)

    return X, y