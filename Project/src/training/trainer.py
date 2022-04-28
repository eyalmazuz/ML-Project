from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef

from ..utils.utils import timeit

class Trainer:
    
    def __init__(self, model, cv, multi_class, **kwargs):
        self.model = model
        self.cv = cv
        self.multi_class = multi_class
        self.kwargs = kwargs

    def train(self,X, y):
        avg_acc, avg_auc, avg_pr, avg_mcc = 0, 0, 0, 0
        avg_train_time, avg_infer_time = 0, 0
        for i, train_index, test_index in enumerate(self.cv.get_split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            _, train_time = timeit(self.model.fit)(X_train, y_train)

            proba, test_time = timeit(self.model.predict_proba)(X_test)
            pred = self.model.predict(X_test)
            
            acc = accuracy_score(y_test, pred)
            if self.multi_class:
                auc = roc_auc_score(y_test, proba, average='micro')
                pr_auc = average_precision_score(y_test, proba, average='micro')

            else:
                auc = roc_auc_score(y_test, proba)
                pr_auc = average_precision_score(y_test, proba)

            mcc = matthews_corrcoef(y_test, pred)

            avg_acc += acc
            avg_auc += auc
            avg_pr += pr_auc
            avg_mcc += mcc

            avg_train_time += train_time
            avg_infer_time += test_time

        avg_acc /= self.cv.get_n_splits()
        avg_auc /= self.cv.get_n_splits()
        avg_pr /= self.cv.get_n_splits()
        avg_mcc /= self.cv.get_n_splits()
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
                    'test_time': avg_infer_time
                }

        return results, stats