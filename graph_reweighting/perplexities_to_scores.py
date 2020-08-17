
import os
import tqdm
import json
import numpy as np
import pandas as pd
from natsort import natsorted, ns

from util import load_df, plot_hist

"""
Loading perplexities
"""


def get_perplexity_from_json(path_to_json):
    """Reads the sentence perplexity outputs from the BERT_LM"""
    with open(path_to_json) as json_file:
        data_json = json.load(json_file)
        result_perplexity = []
        for sentence_dict in data_json:
            perplexity = sentence_dict['ppl']
            result_perplexity.append(perplexity)
    print("result length : ", len(result_perplexity))
    return result_perplexity


def get_perplexity_from_multiple_files(file_dir):
    """
    Reads BERT_LM sentence perplexities from multiple files, e.g. for chunked data.
    Expects subfolders in file_dir, containing a 'test_results.json'.
    get_perplexity_from_multiple_files('BERT_LM_results/')
    """
    list_subfolder = [dir_name for dir_name in os.listdir(file_dir)
                      if os.path.isdir(os.path.join(file_dir, dir_name))]
    sorted_list = natsorted(list_subfolder, alg=ns.IGNORECASE)
    result_list = []
    for fold in sorted_list:
        path_templ = os.path.join(file_dir, fold, 'test_results.json')
        result_list += get_perplexity_from_json(path_templ)
    return result_list


"""
Transforming perplexities to scores
"""


def partial_scale(val, max):
    """Scales all inputs val that are above 1 to [1,50], used in reweight"""
    if val > 1:
        val -= 1
        val = val / (max - 1)
        val = val * 49
        val += 1
    return val


def reweight(ppl):
    scale_good = np.vectorize(partial_scale)

    # Transform perplexities
    ppl = np.log10(ppl)  # log scale against outliers
    max = ppl.max()  # save maximum for keeping cutoff point
    ppl = ppl * -1 + max  # negate and move to [0, ..]
    # (max -2) was x=2 in uninverted log-scale --> equals perplexity of 100
    ppl = ppl / (max - 2)  # rescale so values higher than 100 perplexity are in [0,1]

    scale_max = ppl.max()
    print()
    ppl = scale_good(ppl, scale_max)
    return ppl


def reweight_light(perplex_series, scale_factor):
    ppt = perplex_series
    ppt = (1 / ppt) * scale_factor
    return ppt


"""
Map scores back to the edges of the KG
"""


def apply_reweight(sentence_csv_path, graph_path, perplexity_file_dir, out_file, score_type):
    """
    Reads perplexity results from BERT_LM, applies a reweighting scheme and injects the resulting weights into the KG.
    sentence_csv_path generated in get_all_sentences
    score_type: one of ['reweight', 'reweight_light']
    apply_reweight(sentence_csv_path='cn_sentences.csv', graph_path='conceptnet.csv', perplexity_file_dir='BERT_LM_results/', out_file='cn_reweight.csv', score_type='reweight')
    """
    # Load inputs
    sentence_df = load_df(sentence_csv_path, index=0)
    sentence_idx = sentence_df['0'].tolist()  # only need the indices, not the entire sentences
    perplex = get_perplexity_from_multiple_files(perplexity_file_dir)
    ppt_series = pd.Series(perplex)
    plot_hist(ppt_series, log_scale=True)
    # Load data as dict to increase writing speed
    graph_file = open(graph_path)
    data_dict = {}
    print('Loading data...')
    for i, line in enumerate(graph_file):
        line = line.rstrip()
        data_dict[i] = line.split('\t')
    print('Done.')

    # Convert perplexities to scores
    if score_type == 'reweight':
        ppt_series = reweight(ppt_series)
    elif score_type == 'reweight_light':
        ppt_series = reweight_light(ppt_series, scale_factor=50)
    else:
        raise ValueError(f"score_type needs to be one of ['reweight', 'reweight_light'], but was: {score_type}")
    plot_hist(ppt_series, log_scale=True)  # result for comparison

    # Inject new weights into the KG
    weights = ppt_series.tolist()
    print('Starting injections...')
    injections_num = 0  # checksum
    for idx in tqdm.tqdm(range(len(sentence_idx))):
        injection = data_dict[sentence_idx[idx]]  # row for replacing
        injection[2] = weights[idx]  # replace weight in column 2
        data_dict[sentence_idx[idx]] = injection
        injections_num += 1
    print(injections_num)
    keys = list(data_dict.keys())
    keys.sort()
    results = []
    for i in tqdm.tqdm(range(len(keys))):
        results.append(data_dict[keys[i]])
    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(out_file, sep='\t', index=False, header=False)


if __name__ == '__main__':
    apply_reweight(sentence_csv_path='./output/sentences/cn_toy_sentences.csv',
                   graph_path='./data/examples/conceptnet_toy.csv',
                   perplexity_file_dir='./data/examples/conceptnet_toy_BERT_results/',
                   out_file='./output/kgs/cn_toy_reweight.csv',
                   score_type='reweight')
