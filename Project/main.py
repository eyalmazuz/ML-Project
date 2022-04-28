import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.feature_selection.strategy import StrategyType
from src.training.train_utils import run_experiment

models = [LogisticRegression, SVC, GaussianNB, KNeighborsClassifier, RandomForestClassifier]


# TODO fix this method to be a single run
# and then have a config with all the params needed for all the runs 
def main():
    for (path, label) in [()]:
        for model in models:
            for strategy_type in StrategyType.__members__.values():
                for k in [1,2,3,4,5,10,15,20,25,30,50,100]:
                    params = {'num_features': k, 'discovery_rate': 0.1}
                    run_experiment(path, model, strategy_type, k, params)
                    

if __name__ == '__main__':
    main()