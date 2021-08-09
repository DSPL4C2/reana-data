import sys
import numpy as np
import pandas as pd
from utils import concat
from data_reader import *
from plotting import *
from stats import *
from tabulator import *
from os import mkdir

def main():
    xoffset = 0
    yscale = 'log'
    data_path = 'datasets/data'
    output_path = 'results'
    spls = ['BSN']
    labels = ['Reana', 'ReanaE']

    if len(sys.argv) > 2:
        data_path = sys.argv[1]
        output_path = sys.argv[2]
        spls = sys.argv[3:]
        print(f'data_path: {data_path}')
        print(f'output_path: {output_path}')
        print(f'spls: {spls}')
    try:
        mkdir(output_path)
    except OSError as error:
        pass
    dirs = ['graphs', 'boxplots', 'pairwise-graphs', 'tables', 'tables/effect-size', 'tables/summary']
    for path in dirs:
        try:
            mkdir(f'{output_path}/{path}')
        except OSError as error:
            pass

    # convert data to csv
    rt_data = concat([[f'running_time/totalTime{spl}{label}' for spl in spls] for label in labels])
    mem_data = concat([[f'memory_usage/totalMemory{spl}{label}' for spl in spls] for label in labels])

    for filename in rt_data:
        out_to_csv(f'{data_path}/{filename}.out', f'csv/{filename}.csv')

    for filename in mem_data:
            out_to_csv(f'{data_path}/{filename}.out', f'csv/{filename}.csv')

    for spl in spls:
        plot_spl(spl, labels, xoffset=xoffset, yscale=yscale, output_path=output_path)
        get_pairwise_graphs(spl, labels, xoffset=xoffset, yscale=yscale, output_path=output_path)

def get_pairwise_graphs(spl, labels, xoffset=0, yscale='log', output_path='results'):
   rt_filenames = [
       'csv/running_time/totalTime{}{}.csv'.format(spl, label) for label in labels]
   factor = 1e-3
   rt_df = read_data(spl, rt_filenames, labels, factor=factor)

   # colors = ['darkorange', 'darkviolet', 'royalblue']
   colors = ['tab:blue', 'tab:orange', 'tab:green']
   for items in iter.combinations(zip(labels, colors), 2):
       pair = [item[0] for item in items]
       color = [item[1] for item in items]
       title = '{}: Running Time ({} x {})'.format(spl, pair[0], pair[1])
       xlabel = 'Evolution'
       ylabel = 'Running Time (s)'
       make_line_graph(rt_df, spl, pair, title=title,
                   xlabel=xlabel, ylabel=ylabel, yscale=yscale, xoffset=xoffset,
                   filename=f'{output_path}/pairwise-graphs/{spl}-{pair[0]}-{pair[1]}-rt.png', colors=color)

       mem_filenames = [
           'csv/memory_usage/totalMemory{}{}.csv'.format(spl, label) for label in labels]
       mem_df = read_data(spl, mem_filenames, labels)

       title = '{}: Memory Usage ({} x {})'.format(spl, pair[0], pair[1])
       xlabel = 'Evolution'
       ylabel = 'Memory Usage (MB)'
       make_line_graph(mem_df, spl, pair, title=title,
                   xlabel=xlabel, ylabel=ylabel, yscale=yscale, xoffset=xoffset,
                   filename=f'{output_path}/pairwise-graphs/{spl}-{pair[0]}-{pair[1]}-mem.png', colors=color)

def plot_spl(spl, labels, xoffset=0, yscale='log', output_path='results'):
    rt_filenames = [
        'csv/running_time/totalTime{}{}.csv'.format(spl, label) for label in labels]
    factor = 1e-3
    rt_df = read_data(spl, rt_filenames, labels, factor=factor)
    process(rt_df, spl, labels, '{}: Running Time'.format(spl), header='Running Time',
            ylabel='Running Time (s)', yscale=yscale, xoffset=xoffset, factor=1.0/1000.0, file_suffix='rt',
            table_description="Average running time (s) (statistically smallest value in bold)",
            output_path=output_path)

    mem_filenames = [
        'csv/memory_usage/totalMemory{}{}.csv'.format(spl, label) for label in labels]
    mem_df = read_data(spl, mem_filenames, labels)
    process(mem_df, spl, labels, '{}: Memory Usage'.format(spl), header='Memory Usage',
            ylabel='Memory Usage (MB)', yscale=yscale, xoffset=xoffset, file_suffix='mem',
            table_description="Average memory usage (MB) (statistically smallest value in bold)",
            output_path=output_path)

    l1 = labels[0]
    l2 = labels[1]
    l3 = labels[2] if len(labels) > 2 else None

    rt_df = read_data(spl, rt_filenames, labels,
                      factor=factor, trim_columns=False)
    mem_df = read_data(spl, mem_filenames, labels, trim_columns=False)
    test_df = get_test_comparison_dfs(
        spl, rt_df, mem_df, l1, l2, l3=l3, suffix1='Runtime (s)', suffix2='Memory Usage (MB)', errors1=True, errors2=False, idx_offset=xoffset)

    # write the effect size table to markdown and latex files
    pd.set_option('precision', 2)
    with open(f'{output_path}/tables/effect-size/{spl}.md', 'w') as f:
        test_df = get_test_comparison_dfs(spl, rt_df, mem_df, l1, l2, l3=l3, suffix1='Runtime (s)',
                                          suffix2='Memory Usage (MB)', errors1=True, errors2=False, formatting='markdown', idx_offset=xoffset)
        f.write(test_df.to_markdown(index=False))
    with open(f'{output_path}/tables/effect-size/{spl}.tex', 'w') as f:
        test_df = get_test_comparison_dfs(spl, rt_df, mem_df, l1, l2, l3=l3, suffix1='Runtime (s)',
                                          suffix2='Memory Usage (MB)', errors1=True, errors2=False, formatting='latex', idx_offset=xoffset)
        f.write(test_df.to_latex(index=False, escape=False))

    # write the summary size table to markdown and latex files
    rt_df = read_data(spl, rt_filenames, labels,
                      factor=factor, trim_columns=False)
    mem_df = read_data(spl, mem_filenames, labels, trim_columns=False)
    pd.set_option('precision', 2)
    summary_df = get_summary_dfs(spl, rt_df, mem_df, labels, suffix1='Runtime (s)',
                                 suffix2='Memory Usage (MB)', idx_offset=xoffset)
    with open(f'{output_path}/tables/summary/{spl}.md', 'w') as f:
        f.write(summary_df.to_markdown(index=False))
    with open(f'{output_path}/tables/summary/{spl}.tex', 'w') as f:
        f.write(summary_df.to_latex(index=False))

def process(df, spl, labels, title, header=None, xlabel='Evolution',
            ylabel='', yscale='log', xoffset=1, factor=1.0,
            file_suffix='', table_description=None, output_path='results'):
    make_line_graph(df, spl, labels, title=title,
                    xlabel=xlabel, ylabel=ylabel, yscale=yscale, xoffset=xoffset,
                    filename=f'{output_path}/graphs/{spl}{file_suffix}.png')

    test_results = test_all_evolutions_pairs(df, labels)

    make_box_plot(df, spl, title=title, filename=f'{output_path}/boxplots/{spl}{file_suffix}.png')

    orderings = get_orderings(test_results)

    # mins = [test_min(ordering) for ordering in orderings]
    # remove effect size test from tuples for getting mins
    mins = [test_min([(t1, t2, c) for (t1, t2, c, _) in ordering])
            for ordering in orderings]

    n = get_num_evolutions(df)

    items_per_row = min(n, 10)

    table = get_table(df, labels, title=None,
                      items_per_row=items_per_row, bolded=mins)

    with open(f'{output_path}/tables/{spl}{file_suffix}.md', 'w') as f:
        f.write(table)

if __name__ == '__main__':
    main()
