import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from src.feature_selection.strategy import StrategyType, get_strategy
from src.training.train_utils import run_experiment, run_preprocessing
from src.utils.io import load_data, DataType
from src.utils.utils import timeit
from src.utils.config import configs

PATH = './data/Microbiome'
MODELS = [LogisticRegression, GaussianNB, KNeighborsClassifier, RandomForestClassifier]
SAVE_PATH = './data/logs'
N_FEATURES = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
N_ESTIMATORS = [100, 200, 500, 1000, 5000, 10000]

def main():

    for dataset in os.listdir(PATH):
        try:
            path = os.path.join(PATH, dataset)

            #print(path)
            #print('Loading Data')
            df = load_data(path, DataType.MICROBIOME)

            print('Running preprocessing')
            X, y = run_preprocessing(df)

            if 'SLURM_ARRAY_TASK_ID' in os.environ:
                #print('yay in the right if')

                config_number = int(os.environ['SLURM_ARRAY_TASK_ID'])
                kwargs = {
                    'num_features': configs[config_number][1],
                    'num_estimators': configs[config_number][2],
                    'false_discovery_rate': 0.1,
                }
                model = configs[config_number][0]
                strategy = configs[config_number][3]

                #print(path, kwargs, model, strategy)

                print('Running feature selection')
                strategy = get_strategy(strategy, **kwargs)
                features_mask, feature_selection_time = timeit(strategy.select_features)(X, y)
                if features_mask.sum() != 0:
                    X = X.loc[:, features_mask]
                else:
                    print('Tried to remove all features, keeping original number')

                save_path = os.path.join(SAVE_PATH, f'{os.environ["SLURM_ARRAY_TASK_ID"]}_Model_{model.__name__}_Strategy_{str(strategy)}_Num_Features{kwargs["num_features"]}_Num_Estimators_{kwargs["num_estimators"]}.csv')

                run_experiment(model, X, y, save_path,
                            feature_selection_time=feature_selection_time,
                            num_features_selected=features_mask.sum(),
                            strategy=strategy,
                            original_number_of_features=len(features_mask),
                            dataset_name=dataset)

            else:
                for num_features in N_FEATURES:
                    for num_estimators in N_ESTIMATORS:
                        for strategy in StrategyType:

                            print('Running feature selection')
                            strategy = get_strategy(strategy, num_features=num_features, num_estimators=num_estimators, false_discovery_rate=0.1)
                            features_mask, feature_selection_time = timeit(strategy.select_features)(X, y)
                            X_selected = X.loc[:, features_mask]

                            for model in MODELS:
                                print(f'{dataset=} {num_features=} {num_estimators=} {model=} {strategy=}')
                                run_experiment(model, X_selected, y, SAVE_PATH,
                                            feature_selection_time=feature_selection_time,
                                            num_features_selected=features_mask.sum(),
                                            strategy=str(strategy),
                                            original_number_of_features=len(features_mask),
                                            dataset_name=dataset)
        except:
            pass

    # kwargs = {'num_features': 5, 'false_discovery_rate': 0.1, 'num_estimators': 100, 'data_type': DataType.MAT}
    # model = LogisticRegression()
    # strategy_type = StrategyType.SELECT_FDR

    # run_experiment(model, path, strategy_type, save_path, **kwargs)


if __name__ == '__main__':
    main()
