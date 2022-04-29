import pandas as pd
from sklearn import feature_selection, multiclass

from ..feature_selection.strategy import get_strategy
from ..preprocessing.preprocessing import get_pipeline
from ..cross_validation.cross_validation import get_cross_validation_type
from .trainer import Trainer
from ..utils.utils import log_results, timeit

def run_preprocessing(path, label):
    df = pd.read_csv(path)                         
    # TODO preprocess the data according to the actual project
    # TODO return if the task is multiclass or not

    pipeline = get_pipeline()

    y = df[label]
    X = df.drop(label, axis=1)

    X = pipeline.fit_transform(X)

    return X, y

def run_experiment(model, path, label, strategy_type, save_path, **kwargs):
    strategy = get_strategy(strategy_type, **kwargs)

    print('Running preprocessing')
    X, y = run_preprocessing(path, label)

    print('Running feature selection')
    features_mask, feature_selection_time = timeit(strategy.select_features)(X, y)
    X = X.loc[:, features_mask]

    cv = get_cross_validation_type(X.shape[0])
    # TODO change here to actual logic
    multiclass = True
    print('Training')
    trainer = Trainer(model, cv, multiclass)
    results, stats = trainer.train(X, y)

    print('Logging results')
    log_results(save_path, model, strategy_type, results,
                num_features_selected=features_mask.sum(),
                cv_method=str(cv),
                original_number_of_features=len(features_mask),
                number_of_samples=X.shape[0],
                feature_selection_time=feature_selection_time,
                **kwargs, **stats)