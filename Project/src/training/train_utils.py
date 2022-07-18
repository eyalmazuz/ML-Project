import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..feature_selection.strategy import get_strategy
from ..preprocessing.preprocessing import get_pipeline
from ..cross_validation.cross_validation import get_cross_validation_type
from .trainer import Trainer
from ..utils.utils import log_results, timeit
from ..utils.io import load_data

def run_preprocessing(df):
    # TODO preprocess the data according to the actual project
    # TODO return if the task is multiclass or not

    pipeline = get_pipeline()

    y = df[df.columns[-1]]
    X = df.drop(df.columns[-1], axis=1)

    if y.hasnans:
        y = y.fillna(y.mode().iloc[0])

    #print(df)
    #print(X.shape, y.shape)

    X = pipeline.fit_transform(X)
    y[:] = LabelEncoder().fit_transform(y)
    y = y.astype(int)

    return X, y

def run_experiment(model, X, y, save_path, **kwargs):

    cv = get_cross_validation_type(X.shape[0])
    # TODO change here to actual logic
    print('Training')
    trainer = Trainer(model, cv, y.nunique() > 2)
    results, stats = trainer.train(X, y)

    print('Logging results')
    #print(stats)
    #print(kwargs)
    log_results(save_path, model, results,
                cv_method=str(cv),
                number_of_samples=X.shape[0],
                **kwargs, **stats)
