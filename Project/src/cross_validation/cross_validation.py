from sklearn.model_selection import LeavePOut, LeaveOneOut, KFold

def get_cross_validation_type(df_size: int):
    """
    Get the cross validation type based on the size of the dataframe.
    :param df_size: The size of the dataframe.
    :return: The cross validation type.
    """
    if df_size < 50:
        cv =  LeavePOut(2)

    elif df_size < 100:
        cv = LeaveOneOut()

    elif df_size < 1000:
        cv =  KFold(n_splits=10)
    
    else:
        cv = KFold(n_splits=5)

    return cv