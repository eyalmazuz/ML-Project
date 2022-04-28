import os
import time

import pandas as pd

def log_results(path, model, strategy, k, results, **kwargs):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    di = {}
    di['number of samples'] = kwargs['number_of_samples']
    di['original number of features'] = kwargs['original_number_of_features']
    di['filtering algorithm'] = str(strategy)
    di['model'] = str(model)
    di['number of features selected'] = k
    di['cv method'] = kwargs['cv_method']
    di['cv folds'] = kwargs['cv_folds']
    di['average train time'] = kwargs['train_time']
    di['average test time'] = kwargs['test_time']

    for metric, score in results.items():
        di[metric] = score

    df = df.append(pd.Series(di), ignore_index=True)

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