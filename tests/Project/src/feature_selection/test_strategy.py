import unittest

import pandas as pd

from Project.src.feature_selection.strategy import *

class TestStrategy(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.df = pd.read_csv('./data/MiscRegular/GDS4824f.csv')

    def test_get_strategy(self):
        strategy = get_strategy(StrategyType.SELECT_FDR, **{'num_features': 5})
        self.assertIsInstance(strategy, FdrStrategy)
        self.assertEqual(str(strategy), 'FdrStrategy')

    def test_fdr_strategy(self):
        strategy = FdrStrategy(**{'discovery_rate': 0.1})
        mask = strategy.select_features(self.df.drop('class', axis=1), self.df['class'])
        self.assertEqual(mask.sum(), 46)

    def test_shap_strategy(self):
        strategy = ShapStrategy(**{'num_estimators': 10, 'num_features': 5})
        mask = strategy.select_features(self.df.drop('class', axis=1), self.df['class'])
        self.assertEqual(mask.sum(), 5)

    def test_tree_strategy(self):
        strategy = TreeStrategy(**{'num_estimators': 10, 'num_features': 5})
        mask = strategy.select_features(self.df.drop('class', axis=1), self.df['class'])
        self.assertEqual(mask.sum(), 5)

    def test_rfe_strategy(self):
        strategy = RFEStrategy(**{'num_features': 5})
        mask = strategy.select_features(self.df.drop('class', axis=1), self.df['class'])
        self.assertEqual(mask.sum(), 5)    

    def test_get_strategy_with_invalid_strategy(self):
        with self.assertRaises(ValueError):
            get_strategy('INVALID', **{'num_features': 5})
