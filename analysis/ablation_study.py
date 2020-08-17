
from util import get_en_idx


def get_subgraph(graph_df, source_name, out_template):
    """
    Get subgraph with source_name and write to out_template
    get_subgraph(graph=conceptnet, names=['/d/wiktionary/', '/d/wordnet/'])
    """
    subgraph = graph_df[graph_df['sources'].str.contains(source_name)]
    name = source_name.split('/')[2]
    g_size = subgraph.shape
    subgraph.to_csv(out_template.format(name, str(g_size[0])), sep='\t', index=False, header=False, encoding="utf-8")


def get_pruned_graph(graph_df, threshold, out_template):
    """
    Set all relations with score <= threshold to 0 on all relation that REWEIGHT changes.
    Used for ablation study on reweighted graph.
    get_pruned_graph(graph_df=conceptnet, threshold=20, out_template='cn_new_threshold_{}.csv')
    """
    pattern = "\/c\/en\/"
    graph_df.loc[((graph_df.score <= threshold) & ((~graph_df['relation'].str.contains('/r/dbpedia')) & (
        ~graph_df['relation'].str.contains('/r/Entails'))) & (graph_df['word1'].str.contains(pattern)) & (
                      graph_df['word2'].str.contains(pattern))), 'score'] = 0
    graph_df.to_csv(out_template.format(threshold),
                    sep='\t', index=False, header=False, encoding="utf-8")


def get_original_pruned_graph(original_graph_df, weighted_graph_df, threshold, out_name):
    """
    Set all relation scores in the original graph to 0 that would have score <= threshold in the weighted version
    get_pruned_graph(original_graph_df=cn_orig, weighted_graph_df=cn_new, threshold=20, out_template='cn_orig_threshold_{}.csv')
    """
    pattern = "\/c\/en\/"
    original_graph_df.loc[
        ((weighted_graph_df.score >= threshold) & ((~weighted_graph_df['relation'].str.contains('/r/dbpedia'))
                                                   & (~weighted_graph_df['relation'].str.contains('/r/Entails'))) &
         (weighted_graph_df['word1'].str.contains(pattern))
         & (weighted_graph_df['word2'].str.contains(pattern))), 'score'] = 0
    original_graph_df.to_csv('./Thresholds/{}_{}.csv'.format(out_name, threshold),
                             sep='\t', index=False, header=False, encoding="utf-8")
    print('Prunning Done')


def get_shuffled_graph(graph_df, cols_to_shuffle, out_name, seed):
    """
    Destroys graph by shuffling cols_to_shuffle only, for checking if improvements are better than random
    get_shuffled_graph(graph_df=cn, cols_to_shuffle=['word2'],'cn_new_shuffled.csv')
    """
    shuffled = graph_df[cols_to_shuffle].sample(frac=1, random_state=seed)
    shuffled = shuffled.reset_index(drop=True)
    graph_df[cols_to_shuffle] = shuffled
    graph_df.to_csv(out_name, sep='\t', index=False, header=False, encoding="utf-8")


def get_shuffled_english_graph(graph_df, cols_to_shuffle, out_name, seed):
    """
    Destroys English graph by shuffling cols_to_shuffle only, for checking if improvements are better than random
    get_shuffled_graph(graph_df=cn, cols_to_shuffle=['word2'],'cn_en_new_shuffled.csv')
    """
    en_scores = graph_df.iloc[get_en_idx(graph_df)][cols_to_shuffle]
    shuffled = en_scores.sample(frac=1, random_state=seed)
    shuffled = shuffled.reset_index(drop=True).tolist()
    graph_df.iloc[get_en_idx(graph_df), graph_df.columns.get_loc(cols_to_shuffle)] = shuffled
    graph_df.to_csv(out_name, sep='\t', index=False, header=False, encoding="utf-8")
