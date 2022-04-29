import os
import unittest

import pandas as pd

from Project.src.utils.utils import log_results

class TestUtils(unittest.TestCase):



    def tearDown(self) -> None:
        if os.path.exists('./data/log.csv'):
            os.remove('./data/log.csv')

    def test_log_no_df(self):
        self.assertFalse(os.path.exists('./data/log.csv'))

        log_results('./data/log.csv',
                    'model',
                    'strategy',
                    5,
                    {'metric': 0.1},
                    number_of_samples=10,
                    original_number_of_features=10,
                    cv_method='kfold',
                    cv_folds=10,
                    train_time=0.1,
                    test_time=0.1)

        self.assertTrue(os.path.exists('./data/log.csv'))
        df = pd.read_csv('./data/log.csv')
        self.assertEqual(df.shape[0], 1)

    def test_log_with_df(self):
        log_results('./data/log.csv',
                    'model',
                    'strategy',
                    5,
                    {'metric': 0.1},
                    number_of_samples=10,
                    original_number_of_features=10,
                    cv_method='kfold',
                    cv_folds=10,
                    train_time=0.1,
                    test_time=0.1)

        self.assertTrue(os.path.exists('./data/log.csv'))
        df = pd.read_csv('./data/log.csv')
        self.assertEqual(df.shape[0], 1)

        log_results('./data/log.csv',
                    'model',
                    'strategy',
                    5,
                    {'metric': 0.1},
                    number_of_samples=10,
                    original_number_of_features=10,
                    cv_method='kfold',
                    cv_folds=10,
                    train_time=0.1,
                    test_time=0.1)

        self.assertTrue(os.path.exists('./data/log.csv'))
        df = pd.read_csv('./data/log.csv')
        self.assertEqual(df.shape[0], 2)

