import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from utils import concat
from data_reader import *
from plotting import *
from stats import *
from tabulator import *

def plot_spl(spl, labels, start_index=0):
    def process(df, labels, title, header=None, xlabel='Evolution', ylabel='', xticks=None, yscale='log', start_index=start_index, factor=1.0, table_suffix='', table_description=None):
        if header:
            display(Markdown('## {}'.format(header)))
        make_line_graph(df, spl, labels, title=title, xlabel=xlabel, ylabel=ylabel, xticks=xticks, yscale='log')   

        test_results = test_all_evolutions_pairs(df, labels)

        make_box_plot(df, spl, title=title)

        orderings = get_orderings(test_results)

        # mins = [test_min(ordering) for ordering in orderings]
        # remove effect size test from tuples for getting mins
        mins = [test_min([(t1, t2, c) for (t1, t2, c, _) in ordering]) for ordering in orderings]

        n = get_num_evolutions(df)

        items_per_row = min(n, 10)

        if table_description:
            display(Markdown(table_description))
        table = get_table(df, labels, title=None, items_per_row=items_per_row, bolded=mins, index_offset=start_index)

        display(Markdown(table))

        with open('tables/{}{}.md'.format(spl, table_suffix), 'w') as f:
            f.write(table)

    display(Markdown('# {}'.format(spl)))
    rt_filenames = ['csv/running_time/totalTime{}{}.csv'.format(spl, label) for label in labels]
    factor = 1
    rt_df = read_data(spl, rt_filenames, labels, factor=factor, index_offset=start_index)
    
    xticks = np.arange(start_index, rt_df.shape[0]+start_index)
    
    process(rt_df, labels, '{}: Running Time'.format(spl), header='Running Time',
        ylabel='Running Time (s)', xticks=xticks, yscale='linear', factor=factor, table_suffix='rt',
        table_description="Average running time (s) (statistically smallest value in bold)")

    mem_filenames = ['csv/memory_usage/totalMemory{}{}.csv'.format(spl, label) for label in labels]
    mem_df = read_data(spl, mem_filenames, labels, index_offset=start_index)
    
    xticks = np.arange(start_index, mem_df.shape[0]+start_index)
    
    process(mem_df, labels, '{}: Memory Usage'.format(spl), header='Memory Usage',
        ylabel='Memory Usage (MB)', xticks=xticks, table_suffix='mem',
        table_description="Average memory usage (MB) (statistically smallest value in bold)")

    l1 = labels[0]
    l2 = labels[1]

    rt_df = read_data(spl, rt_filenames, labels, factor=factor, trim_columns=True, index_offset=start_index) 
    mem_df = read_data(spl, mem_filenames, labels, trim_columns=True,  index_offset=start_index)
    # test_df = get_test_comparison_df(rt_df, l1, l2, suffix='Runtime (s)', model=spl)
    test_df = get_test_comparison_dfs(spl, rt_df, mem_df, l1, l2, suffix1='Runtime (s)', suffix2='Memory Usage (MB)')
    pd.set_option('precision', 2)
    with open('tables/effect-size/{}.md'.format(spl), 'w') as f:
            f.write(test_df.to_markdown())
    with open('tables/effect-size/{}.tex'.format(spl), 'w') as f:
            f.write(test_df.to_latex())
    display(test_df)