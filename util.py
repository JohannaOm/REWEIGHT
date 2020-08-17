
import os
import io
import pandas as pd
import matplotlib.pyplot as plt


def get_en_idx(dataframe):
    pattern = "\/c\/en\/"
    index = dataframe.index[
        (dataframe['word1'].str.contains(pattern) & dataframe['word2'].str.contains(pattern))].tolist()
    return index


def concatenate_files(files_to_merge, out_path):
    with io.open(out_path, 'w+', encoding='utf8') as outfile:
        for fname in files_to_merge:
            print(fname)
            with open(fname, 'r') as infile:
                for line in infile:
                    line = line.decode('utf-8', 'ignore')
                    outfile.write(line.strip() + '\t' + fname + '\n')
    print('Merge Done')


def load_df(path_to_data_csv, sep="\t", columns=None, index=None, nrows=None):
    """g_orig = load_df(path_orig, columns=['word1', 'word2', 'score', 'sources', 'relation'])"""
    if columns:
        data_df = pd.read_csv(path_to_data_csv, sep, index_col=index, names=columns, nrows=nrows)
        return data_df
    else:
        data_df = pd.read_csv(path_to_data_csv, sep, index_col=index, nrows=nrows)
        return data_df


def plot_hist(input_data, bins=200, log_scale=False, title=None, out_folder=None, cut_title=True):
    """Plots input data histogram"""
    plt.hist(input_data, bins=bins)
    if log_scale:
        plt.yscale('log')
    plt.grid(False)
    if title:
        label = title.split('/')[-1] if cut_title else title
        plt.gca().set(title='Frequency Histogram ' + label, ylabel='Frequency')
    else:
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    if out_folder:
        plt.savefig(os.path.join(out_folder, title + '.png'), dpi=400)
    plt.show()


def word_to_concept(word):
    word = str(word)
    conc = '/c/en/'
    conc = conc + word.replace(' ', '_')
    return conc


def word_to_rel(word):
    word = str(word)
    if word in ['', '-']:
        word = 'is'
    rel = '/r/'
    rel = rel + word.replace(' ', '_')
    return rel
