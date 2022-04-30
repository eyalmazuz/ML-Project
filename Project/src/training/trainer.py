import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from tqdm import tqdm

from ..utils.utils import timeit

class Trainer:
    
    def __init__(self, model, cv, multi_class, **kwargs):
        self.model = model
        self.cv = cv
        self.multi_class = multi_class
        self.kwargs = kwargs

    def train(self, X, y):
        avg_acc, avg_auc, avg_pr, avg_mcc = 0, 0, 0, 0
        avg_train_time, avg_infer_time = 0, 0
        probas, preds, labels = [], [], []
        for i, (train_index, test_index) in (t := tqdm(enumerate(self.cv.split(X, y)))):
            t.set_description(f'Fold: {i + 1}')
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            _, train_time = timeit(self.model.fit)(X_train, y_train)

            proba, test_time = timeit(self.model.predict_proba)(X_test)
            pred = self.model.predict(X_test)

            probas.append(proba)
            preds.append(pred)
            labels.append(y_test.values)

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        probas = np.concatenate(probas)

        avg_acc = accuracy_score(labels, preds)
        if self.multi_class:
            avg_auc = roc_auc_score(labels, probas, average='macro', multi_class='ovr')
            b = np.zeros((labels.size, labels.max()+1))
            b[np.arange(labels.size), labels] = 1
            avg_pr = average_precision_score(b, probas, average='macro')

        else:
            avg_auc = roc_auc_score(labels, probas[:, 1])
            avg_pr = average_precision_score(labels, probas[:, 1])

        avg_mcc = matthews_corrcoef(labels, preds)

        avg_train_time /= self.cv.get_n_splits()
        avg_infer_time /= self.cv.get_n_splits()
        
        results = {
                    'acc': avg_acc,
                    'auc': avg_auc,
                    'pr_auc': avg_pr,
                    'mcc': avg_mcc, 
        }
        stats = {
                    'train_time': avg_train_time,
                    'test_time': avg_infer_time,
                    'cv_folds': self.cv.get_n_splits()
                }

        return results, stats