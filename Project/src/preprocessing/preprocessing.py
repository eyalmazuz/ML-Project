import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

class Imputer(TransformerMixin):
    """
    Imputer transformer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.imputers = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for column in X.columns:
            if X[column].dtype == 'object':
                self.imputers[column] = SimpleImputer(strategy='most_frequent', missing_values=None)
                X_copy[column] = self.imputers[column].fit_transform(X[column].values.reshape(-1, 1))
            
            elif X[column].dtype == 'int64':
                self.imputers[column] = SimpleImputer(strategy='mean')
                X_copy[column] = self.imputers[column].fit_transform(X[column].values.reshape(-1, 1)) 

            elif X[column].dtype == 'float64':
                self.imputers[column] = SimpleImputer(strategy='mean')
                X_copy[column] = self.imputers[column].fit_transform(X[column].values.reshape(-1, 1))

        return X_copy

class Encoder(TransformerMixin):
    """
    Encoder transformer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.encoders = {}

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        for column in X.columns:
            if X[column].dtype == 'object':
                self.encoders[column] = LabelEncoder()
                X_copy[column] = self.encoders[column].fit_transform(X[column].values.reshape(-1, 1))
        return X_copy

class OutlierRemover(TransformerMixin):
    """
    Outlier remover transformer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.variance_threshold = VarianceThreshold(threshold=0)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        self.variance_threshold.fit(X)
        X_copy = X_copy.loc[:, self.variance_threshold.get_support()]
        X_copy[:] = self.variance_threshold.transform(X_copy)
        return X_copy

class Normalizer(TransformerMixin):
    """
    Normalizer transformer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.power_transformer = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[:] = self.power_transformer.fit_transform(X)
        return X_copy

def get_pipeline():
    imputer = Imputer()
    encoder = Encoder()
    outlier_remover = OutlierRemover()
    normalizer = Normalizer()

    pipeline = Pipeline([
                            ('imputer', imputer),
                            ('encoder', encoder),
                            ('outlier_remover', outlier_remover),
                            ('normalizer', normalizer)
                         ])

    return pipeline

def main():
    df = pd.read_csv('../../../data/classification_datasets/labor.csv')

    imputer = Imputer()
    encoder = Encoder()
    outlier_remover = OutlierRemover()
    normalizer = Normalizer()

    pipeline = Pipeline([
                            ('imputer', imputer),
                            ('encoder', encoder),
                            ('outlier_remover', outlier_remover),
                            ('normalizer', normalizer)
                         ])

    y = df['class']
    X = df.drop(['class'], axis=1)

    print(X.shape)
    X_transformed = pipeline.fit_transform(X, y)

    print(X_transformed)


if __name__ == "__main__":
    main()