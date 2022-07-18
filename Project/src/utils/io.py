from enum import Enum
import os
import re

import scipy.io
from scipy.io import arff
import pandas as pd


class DataType(Enum):

    MAT = 0
    ARFF = 1
    DATAMICROARRAY = 2
    BIOCONDUCTOR = 3
    MICROBIOMIC_DATA = 4
    OTU = 5
    CSV = 6
    MAML = 7
    EFFICIENT_FS = 8
    AMPLICON = 9
    MICROBIOME = 10


def load_data(path, data_type):

    if data_type == DataType.MAT:
        return load_mat(path)

    elif data_type == DataType.ARFF:
        return load_arff(path)

    elif data_type == DataType.DATAMICROARRAY:
        return load_data_micro_array(path)

    elif data_type == DataType.BIOCONDUCTOR:
        return load_bioconductor(path)

    elif data_type == DataType.MICROBIOMIC_DATA:
        return load_microbiomic_data(path)

    elif data_type == DataType.OTU:
        return load_otu(path)

    elif data_type == DataType.CSV:
        return load_csv(path)

    elif data_type == DataType.MAML:
        return load_mAML(path)

    elif data_type == DataType.EFFICIENT_FS:
        return load_efficientFS(path)

    elif data_type == DataType.AMPLICON:
        return load_amplicon_megagenome(path)

    elif data_type == DataType.MICROBIOME:
        return load_microbiome(path)

def load_mat(path):
    mat = scipy.io.loadmat(path)

    df = pd.DataFrame(mat['X'])
    df['Y'] = mat['Y']

    return df

def load_arff(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])

    return df

def load_data_micro_array(path):

    files = os.listdir(path)
    if files[0].endswith('_inputs.csv'):
        X = pd.read_csv(os.path.join(path, files[0]), header=None)
        y = pd.read_csv(os.path.join(path, files[1]), header=None, names=['Y'])

    else:
        X = pd.read_csv(os.path.join(path, files[1]), header=None)
        y = pd.read_csv(os.path.join(path, files[0]), header=None, names=['Y'])

    df = pd.concat([X, y], axis=1)

    return df

def load_bioconductor(path):
    df = pd.read_csv(path, header=0, index_col=0)
    df = df.T

    df = pd.concat([df[df.columns[1:]], df[df.columns[0]]], axis=1)
    df.columns = [re.sub(r'[<,>]', 'a', col) if isinstance(col, str) else f'foo_{i}' for i, col in enumerate(df.columns.tolist())]

    return df

def load_microbiomic_data(path):
    df = pd.read_csv(path, header=None, index_col=0)
    df = df.T

    df = pd.concat([df[df.columns[1:]], df[df.columns[0]]], axis=1)

    return df

def load_otu(path):
    data = pd.read_csv(os.path.join(path, 'otutable.txt'), sep='\t', index_col=0).T
    labels = pd.read_csv(os.path.join(path, 'task.txt'), sep='\t', index_col=0)

    df = data.join(labels)

    return df

def load_csv(path):
    df = pd.read_csv(path)

    return df


def load_mAML(path):

    file_name = path.split('/')[-1]
    X = pd.read_csv(os.path.join(path, f'{file_name}.csv'), index_col=0)
    y = pd.read_csv(os.path.join(path, f'{file_name}.mf.csv'), index_col=0)

    df = X.join(y)

    return df


def load_efficientFS(path):

    if path.endswith('.csv'):
        df = pd.read_csv(path)

    elif path.endswith('.mat'):
        mat = scipy.io.loadmat(path)

        df = pd.DataFrame(mat['X'])
        df['Y'] = mat['Y']

    elif path.endswith('.arff'):
        data = arff.loadarff(path)
        df = pd.DataFrame(data[0])

    return df

def load_amplicon_megagenome(path):

    y = pd.read_csv(os.path.join(path, 'sample_data.xls'), sep='\t', index_col=0, usecols=['#SampleID', 'Disease.MESH.ID'])
    data = pd.read_csv(os.path.join(path, 'otu_table.xls'), sep='\t', index_col=0).T

    df = data.join(y)

    return df

def load_microbiome(path):

    data = pd.read_csv(os.path.join(path, 'otutable.txt'), sep='\t', index_col=0).T
    y = pd.read_csv(os.path.join(path, 'task.txt'), sep='\t', index_col=0)

    df = data.join(y)
    df.columns = [re.sub(r'[\[<,>\]]', 'a', col) if isinstance(col, str) else f'foo_{i}' for i, col in enumerate(df.columns.tolist())]


    return df
