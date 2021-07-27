import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from display import *
from utils import concat
from data_reader import *
from plotting import *
from stats import *
from tabulator import *

if __name__ == '__main__':
    main()

def main():
    xoffset = 0
    yscale = 'log'
    data_path = 'datasets/data'
    output_path = 'results'
    spls = ['BSN']
    labels = ['Reana', 'ReanaE']

    # convert data to csv
    rt_data = concat([[f'running_time/totalTime{spl}{label}' for spl in spls] for label in labels])
    mem_data = concat([[f'memory_usage/totalMemory{spl}{label}' for spl in spls] for label in labels])

    for filename in rt_data:
        out_to_csv(f'{data_path}/{filename}.out', f'csv/{filename}.csv')

    for filename in mem_data:
            out_to_csv(f'{data_path}/{filename}.out', f'csv/{filename}.csv')

    plot_spl('BSN', labels, xoffset=xoffset, yscale=yscale, output_path=output_path)
    get_pairwise_graphs('BSN', labels, xoffset=xoffset, yscale=yscale, output_path=output_path)
