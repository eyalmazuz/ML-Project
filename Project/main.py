import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.feature_selection.strategy import Strategy, StrategyType
from src.training.train_utils import run_experiment

models = [LogisticRegression, SVC, GaussianNB, KNeighborsClassifier, RandomForestClassifier]


# TODO fix this method to be a single run
# and then have a config with all the params needed for all the runs 
def main():

    path = './data/classification_datasets/baseball.csv'
    label = 'Hall_of_Fame'
    kwargs = {'num_features': 5, 'false_discovery_rate': 0.1, 'num_estimators': 100}
    model = LogisticRegression()
    strategy_type = StrategyType.RANDOM_FOREST
    save_path = './data/log.csv'

    run_experiment(model, path, label, strategy_type, save_path, **kwargs)

if __name__ == '__main__':
    main()