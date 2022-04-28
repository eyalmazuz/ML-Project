from cProfile import run
import pandas as pd
from sklearn import multiclass

from ..feature_selection.strategy import get_strategy
from ..preprocessing.preprocessing import get_pipeline
from ..cross_validation.cross_validation import get_cross_validation_type
from .trainer import Trainer
from ..utils.utils import log_results

def run_preprocessing(path, label):
    df = pd.read_csv(path)                         
    # TODO preprocess the data according to the actual project
    # TODO return if the task is multiclass or not

    pipeline = get_pipeline()
    df = pipeline.fit_transform(df)
    y = df[label]
    X = df.drop(label, axis=1)

    return X, y, multiclass

def run_experiment(model, path, label, strategy_type, k, save_path, **kwargs):
    strategy = get_strategy(strategy_type, **kwargs)

    X, y = run_preprocessing(path, label)

    features_mask = strategy.select_features(X, y)
    X = X.loc[:, features_mask]

    cv = get_cross_validation_type(X.shape[0])

    trainer = Trainer(model, X, y, cv, multiclass)
    results = trainer.train()

    log_results(save_path, model, strategy_type, k , results, **kwargs)