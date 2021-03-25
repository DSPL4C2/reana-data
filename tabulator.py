import pandas as pd
import numpy as np
from tabulate import tabulate
from utils import get_means_df, get_num_evolutions, isnan
from functools import reduce


def get_table(df, labels, title=None, items_per_row=0, bolded=None):
    # generating lines and headers/separators
    lines = {}
    mdf = get_means_df(df, labels)
    n = get_num_evolutions(df)
    r = ((n // items_per_row) + 1) * items_per_row
    for label in labels:
        lines[label] = get_lines_for_label(
            mdf, label, n, items_per_row, bolded)

    header = format_row([' Type '] + [' ' + str(i) +
                        ' ' for i in range(items_per_row)])
    first_separator = format_row(['---' for i in range(items_per_row + 1)])
    last_separator = format_row(['---' for i in range(items_per_row + 1)])
    separator = (
        lambda k:
            format_row([' --- '] + [' ' + (str(i) if i < n else '---') +
                       ' ' for i in range(items_per_row * k, items_per_row * (k + 1))])
    )
    separators = [first_separator] + \
        [separator(i) for i in range(1, r // items_per_row)] + [last_separator]

    # writing to buffer
    buffer = ''
    if title:
        buffer += '| {} |\n'.format(str(title))
    buffer += header + '\n'

    flag = True
    i = 0
    while flag:
        buffer += separators[i] + '\n'
        for label in labels:
            if not lines[label]:
                flag = False
                break
            line = lines[label].pop(0)
            buffer += line + '\n'
        i += 1
    return buffer


def get_lines_for_label(mdf, label, n, m=0, mins=None):
    '''
    mdf: dataframe containing averages
    label: label for which table lines are generated
    n: number of total elements for each label in mdf
    m: number of elements per row
    mins: label to be bolded for each item
    '''
    lines = []
    r = ((n // m) + 1) * m
    s = '| {} '.format(label)
    for i in range(r):
        if i < n:
            x = mdf.loc[label].iloc[i]
            value = '--' if isnan(x) else '{x:.2f}'.format(x=x)
            if i < n and mins is not None and mins[i] == label:
                s += '| **{}** '.format(value)
            else:
                s += '| {} '.format(value)
        else:
            value = '--'
            s += '| {} '.format(value)

        if m != 0 and i != 0 and i != (r-1) and (i+1) % m == 0:
            lines.append(s + '|')
            s = '| {} '.format(label)

    lines.append(s + '|')
    return lines


def format_row(items):
    return reduce(
        lambda a, b: a + b,
        ['|{}'.format(x) for x in items] + ['|']
    )
