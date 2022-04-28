import time

def log_results(path, model, strategy, k, results, **params):
    pass


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start

        return res, elapsed
    
    return wrapper