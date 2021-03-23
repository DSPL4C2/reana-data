import numpy as np
import pandas as pd
import itertools as iter


def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def isnan(x):
    return x is float('nan') or x == 'nan' or x == np.nan or np.isnan(x)


def concat(lists):
    return list(iter.chain.from_iterable(lists))


def get_means_df(df, labels):
    dfs = [np.mean(df.loc[label]) for label in labels]
    return pd.concat(dfs, keys=labels)


def get_num_evolutions(df):
    label = df['Label'][0]
    return df.loc[label].shape[1] - 1  # exclude type column


def get_evolution_samples(df, n, label):
    s = df.iloc[:][n].loc[label]
    if s.isnull().any():  # check if any sample is NaN
        return None
    else:
        return s
