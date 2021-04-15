import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from utils import concat
from data_reader import *
from plotting import *
from stats import *
from tabulator import *

def get_pairwise_graphs(spl, labels):
   display(Markdown('# {}'.format(spl)))
   rt_filenames = [
       'csv/running_time/totalTime{}{}.csv'.format(spl, label) for label in labels]
   factor = 1e-3
   rt_df = read_data(spl, rt_filenames, labels, factor=factor)    

   colors = ['red', 'green', 'blue']
   for items in iter.combinations(zip(labels, colors), 2):
       pair = [item[0] for item in items]
       color = [item[1] for item in items]
       display(Markdown('## {} x {}'.format(pair[0], pair[1])))
       title = '{}: Running Time ({} x {})'.format(spl, pair[0], pair[1])
       xlabel = 'Evolution'
       ylabel = 'Running Time (s)'
       make_line_graph(rt_df, spl, pair, title=title,
                   xlabel=xlabel, ylabel=ylabel, yscale='log', filename=f'pairwise-graphs/{spl}-{pair[0]}-{pair[1]}.png', colors=color)

       mem_filenames = [
           'csv/memory_usage/totalMemory{}{}.csv'.format(spl, label) for label in labels]
       mem_df = read_data(spl, mem_filenames, labels)

       title = '{}: Memory Usage ({} x {})'.format(spl, pair[0], pair[1])
       xlabel = 'Evolution'
       ylabel = 'Memory Usage (MB)'
       make_line_graph(mem_df, spl, pair, title=title,
                   xlabel=xlabel, ylabel=ylabel, yscale='log', filename=f'pairwise-graphs/{spl}-{pair[0]}-{pair[1]}.png', colors=color)

def plot_spl(spl, labels):
    display(Markdown('# {}'.format(spl)))
    rt_filenames = [
        'csv/running_time/totalTime{}{}.csv'.format(spl, label) for label in labels]
    factor = 1e-3
    rt_df = read_data(spl, rt_filenames, labels, factor=factor)
    process(rt_df, spl, labels, '{}: Running Time'.format(spl), header='Running Time',
            ylabel='Running Time (s)', factor=1.0/1000.0, table_suffix='rt',
            table_description="Average running time (s) (statistically smallest value in bold)")

    mem_filenames = [
        'csv/memory_usage/totalMemory{}{}.csv'.format(spl, label) for label in labels]
    mem_df = read_data(spl, mem_filenames, labels)
    process(mem_df, spl, labels, '{}: Memory Usage'.format(spl), header='Memory Usage',
            ylabel='Memory Usage (MB)', table_suffix='mem',
            table_description="Average memory usage (MB) (statistically smallest value in bold)")

    l1 = labels[0]
    l2 = labels[1]

    rt_df = read_data(spl, rt_filenames, labels,
                      factor=factor, trim_columns=True)
    mem_df = read_data(spl, mem_filenames, labels, trim_columns=True)
    test_df = get_test_comparison_dfs(
        spl, rt_df, mem_df, l1, l2, suffix1='Runtime (s)', suffix2='Memory Usage (MB)', errors1=True, errors2=False)
    display(test_df)
    pd.set_option('precision', 2)
    with open('tables/effect-size/{}.md'.format(spl), 'w') as f:
        test_df = get_test_comparison_dfs(spl, rt_df, mem_df, l1, l2, suffix1='Runtime (s)',
                                          suffix2='Memory Usage (MB)', errors1=True, errors2=False, formatting='markdown')
        f.write(test_df.to_markdown(index=False))
    with open('tables/effect-size/{}.tex'.format(spl), 'w') as f:
        test_df = get_test_comparison_dfs(spl, rt_df, mem_df, l1, l2, suffix1='Runtime (s)',
                                          suffix2='Memory Usage (MB)', errors1=True, errors2=False, formatting='latex')
        f.write(test_df.to_latex(index=False, escape=False))

def process(df, spl, labels, title, header=None, xlabel='Evolution', ylabel='', yscale='log', factor=1.0, table_suffix='', table_description=None):
    if header:
        display(Markdown('## {}'.format(header)))
    make_line_graph(df, spl, labels, title=title,
                    xlabel=xlabel, ylabel=ylabel, yscale='log', filename=f'graphs/{spl}.png')

    test_results = test_all_evolutions_pairs(df, labels)

    make_box_plot(df, spl, title=title, filename=f'boxplots/{spl}.png')

    orderings = get_orderings(test_results)

    # mins = [test_min(ordering) for ordering in orderings]
    # remove effect size test from tuples for getting mins
    mins = [test_min([(t1, t2, c) for (t1, t2, c, _) in ordering])
            for ordering in orderings]

    n = get_num_evolutions(df)

    items_per_row = min(n, 10)

    if table_description:
        display(Markdown(table_description))

    table = get_table(df, labels, title=None,
                      items_per_row=items_per_row, bolded=mins)

    display(Markdown(table))

    with open('tables/{}{}.md'.format(spl, table_suffix), 'w') as f:
        f.write(table)
