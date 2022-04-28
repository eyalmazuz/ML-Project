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

def run_experiment(model, X, y, strategy_type, k, path, params):
    cv = get_cross_validation_type(X.shape[0])

    trainer = Trainer(model, X, y, cv, multiclass)
    results = trainer.train()

    log_results(path, model, strategy_type, k , results, **params)


def run_tree_based_experiment(model, path, label, strategy_type, k, params):
    for n in [100, 200, 500, 1000, 5000, 10000]:
        params['num_estimators'] = n
        strategy = get_strategy(strategy_type, **params)

        X, y = run_preprocessing(path, label)

        features_mask = strategy.select_features(X, y)
        X = X.loc[:, features_mask]

        run_experiment(model, X, y, str(strategy), k, path, params)

def run_fdr_based_experiment(model, path, label, strategy_type, k, params):
    strategy = get_strategy(strategy_type, **params)
    X, y = run_preprocessing(path, label)

    features_mask = strategy.select_features(X, y)
    X = X.loc[:, features_mask]

    run_experiment(model, X, y, str(strategy), k, path, params)


def run_rfe_based_experiment(model, path, label, strategy_type, k, params):
    strategy = get_strategy(strategy_type, **params)
    X, y = run_preprocessing(path, label)

    features_mask = strategy.select_features(X, y)
    X = X.loc[:, features_mask]

    run_experiment(model, X, y, str(strategy), k, path, params)

