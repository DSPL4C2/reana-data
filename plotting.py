import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def make_line_graph(df, spl, labels, title=None,
    yscale='linear', xlabel='x', ylabel='y',
    xoffset=0, filename=None, colors=None):

    means = []
    stds = []

    for label in labels:
        dfl = df.loc[label]
        means.append(np.mean(dfl))
        stds.append(np.std(dfl))

    df_mean = pd.concat(means, keys=labels)
    df_std = pd.concat(stds, keys=labels)

    plt.figure(figsize=(12, 5))

    xticks = np.arange(0, df_mean[labels[0]].shape[0], 1)
    xticks_labels = np.arange(xoffset, df_mean[labels[0]].shape[0] + xoffset, 1) 

    axs = []

    for i in range(len(labels)):
        label = labels[i]
        if i == 0:
            if colors:
                ax = df_mean[label].plot(
                    grid=True, yerr=df_std[label], label=label, color=[colors[i]])
            else:
                ax = df_mean[label].plot(
                    grid=True, yerr=df_std[label], label=label)
        else:
            if colors:
                ax = df_mean[label].plot(
                    grid=True, secondary_y=False, yerr=df_std[label], label=label, color=[colors[i]])
            else:
                ax = df_mean[label].plot(
                    grid=True, secondary_y=False, yerr=df_std[label], label=label) 

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)
        axs.append(ax)

    # plt.legend(h1+h2, l1+l2, loc=2)
    plt.legend()
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        plt.title(spl)
    else:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def make_box_plot(df, spl, title=None, xlabel='', ylabel='', filename=None):
    means = np.mean(df)
    stdevs = np.std(df)
    fig, ax_new = plt.subplots(
        5, 5, sharex=False, sharey=False, figsize=(20, 20))

    # these magics work for 5 by 5 plots, which is appropriate for our current data dimensions
    i = 4
    j = 4
    while i * 5 + j + 1 > len(means):
        fig.delaxes(ax_new[i, j])
        if j == 0:
            j = 4
            i -= 1
        else:
            j -= 1

    bp = df.boxplot(by="Label", ax=ax_new.flatten()
                    [:len(means)], figsize=(6, 8))

    for i in range(len(means)):
        ax_new.flatten()[i].autoscale()
        ax_new.flatten()[i].set_xlabel(xlabel)
        ax_new.flatten()[i].set_ylabel(ylabel)

    if title is None:
        title = spl

    fig.suptitle(title)

    if filename is not None:
        plt.savefig(filename)

    plt.show()