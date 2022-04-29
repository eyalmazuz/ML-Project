import os
import time

import pandas as pd

def log_results(path, model, strategy, results, **kwargs):
    
    di = {}
    di['number of samples'] = kwargs['number_of_samples']
    di['original number of features'] = kwargs['original_number_of_features']
    di['filtering algorithm'] = str(strategy)
    di['model'] = str(model)
    di['feature selection time'] = kwargs['feature_selection_time']
    di['number of features selected'] = kwargs['num_features_selected']
    di['cv method'] = kwargs['cv_method']
    di['cv folds'] = kwargs['cv_folds']
    di['average train time'] = kwargs['train_time']
    di['average test time'] = kwargs['test_time']

    for metric, score in results.items():
        di[metric] = score

    for key, value in di.items():
        di[key] = [value]

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=di.keys())

    df = pd.concat([df, pd.DataFrame.from_dict(di, orient='columns')], axis=0)

    df.to_csv(path, index=False)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start

        if res is not None:
            return res, elapsed
        else:
            return elapsed
    
    return wrapper