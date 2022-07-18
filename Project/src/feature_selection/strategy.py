from abc import ABC, abstractmethod
import enum

import numpy as np
import pandas as pd

from shap import TreeExplainer
from sklearn.feature_selection import RFE, SelectFdr
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class StrategyType(enum.Enum):
    """
    The type of feature selection strategy.
    """
    RANDOM_FOREST = enum.auto()
    SELECT_FDR = enum.auto()
    TREE_EXPLAINER = enum.auto()
    RFE = enum.auto()

class Strategy(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if 'num_estimators' in kwargs:
            self.num_estimators = kwargs['num_estimators']
        else:
            self.num_estimators = 10
        
        if 'num_features' in kwargs:
            self.num_features = kwargs['num_features']
        else:
            self.num_features = 10
        
        if 'false_discovery_rate' in kwargs:
            self.discovery_rate = kwargs['false_discovery_rate']
        else:
            self.discovery_rate = 0.1
    
        self.kwargs = kwargs

    @abstractmethod
    def select_features(self, X, y, **kwargs):
        pass

def get_strategy(strategy_type: StrategyType, **kwargs) -> Strategy:
    """
    Get the feature selection strategy based on the type.
    :param strategy_type: The type of feature selection strategy.
    :return: The feature selection strategy.
    """
    if strategy_type == StrategyType.RFE:
        return RFEStrategy(**kwargs)
    elif strategy_type == StrategyType.SELECT_FDR:
        return FdrStrategy(**kwargs)
    elif strategy_type == StrategyType.RANDOM_FOREST:
        return TreeStrategy(**kwargs)
    elif strategy_type == StrategyType.TREE_EXPLAINER:
        return ShapStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

class RFEStrategy(Strategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_features(self, X, y, **kwargs):
        rf = XGBClassifier(n_estimators=self.num_estimators, n_jobs=-1)
        rf.fit(X, y)
        rfe = RFE(estimator=rf, n_features_to_select=self.num_features, step=0.05)
        rfe.fit(X, y)
        return rfe.get_support()
    
    def __str__(self):
        return 'RFEStrategy'

class FdrStrategy(Strategy):
    
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def select_features(self, X, y, **kwargs):
            fdr = SelectFdr(alpha=self.discovery_rate)
            fdr.fit(X, y)
            return fdr.get_support()

        def __str__(self):
            return 'FdrStrategy'

class ShapStrategy(Strategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_features(self, X, y, **kwargs):

        clf=  XGBClassifier(n_estimators=self.num_estimators, n_jobs=-1)
        clf.fit(X, y)

        te = TreeExplainer(clf, feature_perturbation='interventional')

        feature_importance = te.shap_values(X, y)
        if isinstance(feature_importance, list):
            feature_importance = feature_importance[0]
        
        feature_importance = feature_importance.sum(axis=0)

        mask = np.zeros_like(feature_importance, dtype=bool)
        mask[np.argsort(feature_importance)[-self.num_features:]] = True

        return mask
    def __str__(self):
        return 'ShapStrategy'


class TreeStrategy(Strategy):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def select_features(self, X, y, **kwargs):
        clf = XGBClassifier(n_estimators=self.num_estimators, n_jobs=-1)
        clf.fit(X, y)
        
        feature_importance =  clf.feature_importances_
        mask = np.zeros_like(feature_importance, dtype=bool)
        mask[np.argsort(feature_importance)[-self.num_features:]] = True

        return mask

    def __str__(self):
        return 'TreeStrategy'


def main():

    df = pd.read_csv('../../../data/classification_datasets/acute-nephritis.csv')
    y = df['clase']
    X = df.drop('clase', axis=1)

    strategy = get_strategy(StrategyType.SELECT_FDR, **{'num_features': 5})
    mask = strategy.select_features(X, y)
    print(mask)
    print(X.shape)
    print(X.loc[:, mask].shape)
if __name__ == '__main__':
    main()
