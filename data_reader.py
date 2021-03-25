import pandas as pd
import numpy as np
from utils import isNumber


def out_to_csv(in_filename, out_filename):
    data = []
    i = -1
    with open(in_filename) as in_f:
        lines = in_f.readlines()
    for line in lines:
        if not isNumber(line):
            i += 1
            data.append([])
        else:
            data[i].append(float(line))
    with open(out_filename, 'w') as out_f:
        for i in range(len(data)):
            for j in range(len(data[i])):
                if j == len(data[i])-1:
                    out_f.write("{}\n".format(data[i][j]))
                else:
                    out_f.write("{},".format(data[i][j]))


def _read_data(spl, filenames, labels, factor=1.0, trim_rows=True, trim_columns=False):
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename, header=None).transpose()
        dfs.append(df)

    if trim_rows:
        # delete extra rows (to match sample sizes)
        n = min([df.shape[0] for df in dfs])
        for df in dfs:
            if df.shape[0] > n:
                rows = list(range(n, df.shape[0]))
                df = df.drop(rows, inplace=True)

    if trim_columns:
        # delete extra columns (to match number of evolutions)
        m = min([df.shape[1] for df in dfs])
        for df in dfs:
            if df.shape[1] > m:
                cols = list(range(m, df.shape[1]))
                df = df.drop(cols, axis=1, inplace=True)

    df = pd.concat(dfs, keys=labels)
    df *= factor

    return df, n


def read_data(spl, filenames, labels, factor=1.0, trim_rows=True, trim_columns=False):
    df, n = _read_data(spl, filenames, labels, factor=factor,
                       trim_rows=trim_rows, trim_columns=trim_columns)
    df['Label'] = np.repeat(labels, np.repeat([n], len(labels)), axis=0)
    return df
