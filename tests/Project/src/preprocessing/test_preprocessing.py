import unittest

import numpy as np
import pandas as pd

from Project.src.preprocessing.preprocessing import Imputer, Encoder, OutlierRemover

class TestImputer(unittest.TestCase):

    def test_no_missing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        self.assertEqual(df.isna().sum().sum(), 0)
        imputer = Imputer()
        transformed_df = imputer.fit_transform(df)
        self.assertEqual(df.isna().sum().sum(), 0)
        self.assertEqual(transformed_df.isna().sum().sum(), df.isna().sum().sum())
        print(df.isna().sum().sum())


    def test_missing_values_float(self):
        df = pd.DataFrame({'a': [1., None, 3.], 'b': [4., None, 6.]})

        self.assertEqual(df.isna().sum().sum(), 2)
        imputer = Imputer()
        transformed_df = imputer.fit_transform(df)
        self.assertEqual(transformed_df.isna().sum().sum(), 0)

        self.assertEqual(transformed_df.iloc[1, 0], 2.)
        self.assertEqual(transformed_df.iloc[1, 1], 5.)


    def test_missing_values_categorical(self):
        df = pd.DataFrame({'a': ['a', None, 'a', 'b'], 'b': ['a', None, 'b', 'c']})
        self.assertEqual(df.isna().sum().sum(), 2)
        imputer = Imputer()
        transformed_df = imputer.fit_transform(df)
        self.assertEqual(transformed_df.isna().sum().sum(), 0)

        self.assertEqual(transformed_df.iloc[1, 0], 'a')
        self.assertEqual(transformed_df.iloc[1, 1], 'a')

    def test_missing_values_both(self):
        df = pd.DataFrame({'a': [1., None, 3.], 'b': ['a', None, 'b']})
        self.assertEqual(df.isna().sum().sum(), 2)
        imputer = Imputer()
        transformed_df = imputer.fit_transform(df)
        self.assertEqual(transformed_df.isna().sum().sum(), 0)

        self.assertEqual(transformed_df.iloc[1, 0], 2)
        self.assertEqual(transformed_df.iloc[1, 1], 'a')

class TestEncoder(unittest.TestCase):
        
        def test_all_numbers(self): 
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            encoder = Encoder()
            transformed_df = encoder.fit_transform(df)

            self.assertEqual(len(encoder.encoders), 0)


        def test_all_categorical(self):
            df = pd.DataFrame({'a': ['a', 'b', 'c'], 'b': ['a', 'b', 'c']})
            encoder = Encoder()
            transformed_df = encoder.fit_transform(df)

            self.assertEqual(len(encoder.encoders), 2)

            class_0 = encoder.encoders['a'].classes_[0]
            loc = transformed_df[transformed_df['a'] == 0].index.tolist()[0]
            print(encoder.encoders['a'].classes_, class_0)
            self.assertEqual(df.loc[loc, 'a'] , class_0)

class TestOutlierRemover(unittest.TestCase):
    
        def test_no_outliers(self):
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            outlier_remover = OutlierRemover()
            transformed_df = outlier_remover.fit_transform(df)
            self.assertEqual(transformed_df.shape, (3, 2))

        def test_outliers(self):
            df = pd.DataFrame({'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'b': [1, 5, 6, 7, 5, 9, 10, 11, 12, 13]})
            outlier_remover = OutlierRemover()
            transformed_df = outlier_remover.fit_transform(df)
            self.assertEqual(transformed_df.shape, (10, 1))
            
class TestNormalizer(unittest.TestCase):
    pass