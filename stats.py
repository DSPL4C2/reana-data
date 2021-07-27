import pandas as pd
import numpy as np
import itertools as iter
from utils import concat, get_num_evolutions, get_evolution_samples, isnan
from math import sqrt
from scipy.stats import bartlett, ttest_ind, mannwhitneyu, normaltest
from sigfigs import format_value, format_value_error


def compare_samples(s1, s2, significance=0.01):
    if s1.equals(s2):
        return {'result': 'eq', 'method': 'all-equal', 'd': 0}
    if (is_normally_distributed(s1, significance=significance)
            and is_normally_distributed(s2, significance=significance)):
        # both normally distributed
        # test variances
        v1, p1 = bartlett(s1, s2)
        equal_var = p1 >= significance

        # equal variances is assumed unless stated otherwise
        v2, p2 = ttest_ind(s1, s2, equal_var=equal_var)

        method = 'T-test' if equal_var else 'Welch'      # from ttest_ind's documentation
        eq = p2 >= significance
        diff = np.mean(s1) - np.mean(s2)
        r = "eq" if eq else 'gt' if diff > 0 else 'lt'
        d = 0.0 if r == 'eq' else cohens_d(s1, s2)
        return {'result': r, 'method': method, 'p1': p1, 'v1': v1, 'p2': p2, 'v2': v2, 'd': d}
    else:
        # not both normally distributed
        u, p = mannwhitneyu(s1, s2, use_continuity=False)
        eq = p >= significance
        diff = np.mean(s1) - np.mean(s2)
        r = 'eq' if eq else 'gt' if diff > 0 else 'lt'
        d = 0.0 if r == 'eq' else cliffs_delta_from_mann_whitney(
            u, len(s1), len(s2))
        return {'result': r, 'method': 'Mann-Whitney', 'u': u, 'p2': p, 'd': d}


def test_all_evolutions_pairs(df, labels):
    pairs = list(iter.combinations(labels, 2))
    return [(t1, t2, test_all_evolutions(df, t1, t2)) for (t1, t2) in pairs]


def test_all_evolutions(df, t1, t2):
    n = get_num_evolutions(df)
    return [test(df, i, t1, t2) for i in range(n)]


def test(df, i, t1, t2):
    s1 = get_evolution_samples(df, i, label=t1)
    s2 = get_evolution_samples(df, i, label=t2)
    if s1 is None or s2 is None:
        # one of the samples contains NaN
        return None
    else:
        return compare_samples(s1, s2)


def test_min(ordering):
    '''
    ordering has format
    [(x, y, compare(x,y)), (x, z, compare(x,z)), (y, z, compare(y,z))]
    returns the smallest element or None if there is no such element.

    Example: [(x, y, 'eq'), (x, z, 'gt'), (y, z, 'gt')] => z
    '''
    candidates = get_valid_labels(ordering)
    for (t1, t2, c) in ordering:
        if c == 'eq':
            if t1 in candidates:
                candidates.remove(t1)
            if t2 in candidates:
                candidates.remove(t2)
        elif c == 'lt':
            if t2 in candidates:
                candidates.remove(t2)
        elif c == 'gt':
            if t1 in candidates:
                candidates.remove(t1)
    if len(candidates) == 1:
        return candidates[0]
    else:
        return None


def get_orderings(tests):
    orderings = (
        list(
            zip(*[
                [
                    (t1, t2, t['result'] if t else None, t['d'] if t else None) for t in ts
                ]
                for (t1, t2, ts) in tests]
                )
        )
    )
    return orderings


def get_valid_labels(ordering):
    elements = []
    for element in ordering:
        if element[2] is not None:
            elements.append(element[0:2])
    return list(set(concat(elements)))

def get_summary_dfs(model, df1, df2, labels, suffix1=None, suffix2=None, idx_offset=0):
    summary_df1 = get_summary_df(df1, labels, suffix=suffix1)
    summary_df2 = get_summary_df(df2, labels, suffix=suffix2)

    n1 = get_num_evolutions(df1)
    n2 = get_num_evolutions(df2)

    if (n1 != n2):
        print("Warning: number of evolutions in df1 and df2 do not match")

    model_labels = ['{} {}'.format(model, i) for i in range(idx_offset, n1 + idx_offset)]
    model_df = pd.DataFrame({'Model': model_labels})

    return pd.concat([model_df, summary_df1, summary_df2], axis=1)

def get_summary_df(df, labels, suffix=None):
    data = {}
    new_labels = [label if suffix is None else '{} {}'.format(label, suffix) for label in labels]
    for (label, new_label) in zip(labels, new_labels):
        avg = np.mean(df.loc[label])
        std = np.std(df.loc[label])
        lines = []
        for i in range(len(avg)):
            line = format_value_error(avg[i], std[i])
            lines.append(line)
        data[new_label] = lines

    return pd.DataFrame(data=data)


def process_effect_size(d):
    d = abs(d)
    points = [('--', 0.0), ('Small', 0.2), ('Medium', 0.5), ('Large', 0.8)]
    dists = [(label, abs(d - x)) for (label, x) in points]
    dists.sort(key=lambda item: item[1])
    return dists[0][0]


def get_test_comparison_df(df, l1, l2, l3=None, suffix=None, errors=True, formatting=None):
    tests = test_all_evolutions(df, l1, l2)
    comparisons = [test['result'] for test in tests if test is not None]

    [avg1, avg2] = [np.mean(df.loc[l]) for l in [l1, l2]]
    [std1, std2] = [np.std(df.loc[l]) for l in [l1, l2]]

    effect_size = [process_effect_size(test['d']) if test else None
                   for test in tests]
    # hypothesis: assume that both labels obtain the same result
    hypothesis_results = ['Not Reject' if x ==
                          'eq' else 'Reject' for x in comparisons]

    [label1, label2] = [l if suffix is None else '{} {}'.format(l, suffix) for l in [l1, l2]]

    lines1 = []
    lines2 = []
    lines3 = []

    if l3:
        avg3 = np.mean(df.loc[l3])
        std3 = np.std(df.loc[l3])
        label3 = l3 if suffix is None else '{} {}'.format(l3, suffix)

    n = get_num_evolutions(df)
    for i in range(0, n):
        if errors:
            line1 = format_value_error(avg1[i], std1[i])
            line2 = format_value_error(avg2[i], std2[i])
            line3 = format_value_error(avg3[i], std3[i]) if l3 else None
        else:
            line1 = '--' if isnan(avg1[i]) else '{:.2f}'.format(avg1[i])
            line2 = '--' if isnan(avg2[i]) else '{:.2f}'.format(avg2[i])
            if l3:
                line3 = '--' if isnan(avg3[i]) else '{:.2f}'.format(avg3[i])

        # we don't have to format line3 since it is not in the comparison
        if formatting == 'markdown' and i < len(comparisons):
            line1 = f'**{line1}**' if comparisons[i] == 'lt' else line1
            line2 = f'**{line2}**' if comparisons[i] == 'gt' else line2

        elif formatting == 'latex' and i < len(comparisons):
            line1 = '\\textbf{{{}}}'.format(line1) if comparisons[i] == 'lt' else line1
            line2 = '\\textbf{{{}}}'.format(line2) if comparisons[i] == 'gt' else line2

        lines1.append(line1)
        lines2.append(line2)
        lines3.append(line3) if l3 else None

    effect_size = [es if es else '--' for es in effect_size]

    if l3:
        data = {label1: lines1, label2: lines2, label3: lines3, 'Effect Size': effect_size}
    else:
        data = {label1: lines1, label2: lines2, 'Effect Size': effect_size}

    return pd.DataFrame(data=data)

def get_test_comparison_dfs(model, df1, df2, l1, l2, l3=None, suffix1=None, suffix2=None, errors1=True, errors2=True, formatting=None, idx_offset=0):
    test_df1 = get_test_comparison_df(df1, l1, l2, l3=l3, suffix=suffix1, errors=errors1, formatting=formatting)
    test_df2 = get_test_comparison_df(df2, l1, l2, l3=l3, suffix=suffix2, errors=errors2, formatting=formatting)

    n1 = get_num_evolutions(df1)
    n2 = get_num_evolutions(df2)

    if (n1 != n2):
        print("Warning: number of evolutions in df1 and df2 do not match")

    model_labels = ['{} {}'.format(model, i) for i in range(idx_offset, n1 + idx_offset)]
    model_df = pd.DataFrame({'Model': model_labels})

    return pd.concat([model_df, test_df1, test_df2], axis=1)

def is_normally_distributed(samples, significance=0.01):
    x, p = normaltest(samples)
    return p >= significance


def cohens_d(s1, s2):
    assert(len(s1) == len(s2))
    N = len(s1)

    if N < 50:
        factor = ((N-3)/(N-2.25)) * sqrt((N-2)/N)
    else:
        factor = 1

    s = factor * sqrt((np.std(s1, ddof=1)**2 + np.std(s2, ddof=1)**2) / 2)
    cohens_d = (np.mean(s1) - np.mean(s2)) / s
    return cohens_d


def cliffs_delta_from_mann_whitney(u, m, n):
    return 2*u / (m*n) - 1
