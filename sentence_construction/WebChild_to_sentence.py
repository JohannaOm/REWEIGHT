
import pandas as pd

from sentence_construction.graph_to_sentence import concept_to_word
from util import load_df, concatenate_files, word_to_concept, word_to_rel

"""
Get other graphs to same format as ConceptNet
Example: WebChild
"""

webchild_relation_type_to_sentence = {'/r/time': 'one can {} during a {}',
                                      '/r/location': 'one can {} in a {}',
                                      '/r/emotion': 'to {} causes {}',
                                      '/r/agent': 'a {} can {}',
                                      '/r/activity': 'to {} one needs to {}',
                                      '/r/participant': 'a {} is a part of {}',
                                      '/r/thing': 'to {} is related to a {}',
                                      '/r/hassynsetmember': 'to {} is similar to a {}',
                                      '/r/next': 'to {} follows {}',
                                      '/r/prev': 'to {} follows {}',
                                      '/r/hassimilar': 'to {} is similar to {}',
                                      '/r/hashypernymy': 'to {} is to {}',
                                      '/r/hasMember': 'a {} is a member of a {}',
                                      '/r/hasPart': 'a {} is a part of a {}',
                                      '/r/hasSubstance': 'a {} consists of a {}',
                                      '/r/IsA': 'a {} is a {}'
                                      }

"""
Put WebChild into ConceptNet format

Changes for WebChild:
-Words need to be mapped to concepts '/c/en/...'
-WebChild subgraphs dont have consistent formatting -> special extraction method for some subgraphs
-WebChild has subgraphs with "free relations" (no fixed relation types) -> mark with 'comparative' source type
-WebChild can contain multiple identical relation triples -> take triple with highest score only
-Sentence construction needs changes:
    -Free relations on subgraphs -> fit relation name directly into sentence
    -Relation 'IsA' already contains the 'a' -> don't add additionally
    -On some relations concepts have to be flipped when mapping to the proposed sentence templates
"""


def webchild_to_conceptnet_format(graph_df, out_path):
    """
    webchild_to_conceptnet_format(graph_df=webchild_df, out_path='wc_cnformat.csv')
    """
    assertions_temp_list = []
    i = 0
    for index, row in graph_df.iterrows():
        i += 1
        w1 = str(row['#x'])
        w2 = str(row['y'])
        if w1.find('#') != -1:
            w1 = w1[:w1.find('#')]
        if w2.find('#') != -1:
            w2 = w2[:w2.find('#')]
        c1 = word_to_concept(w1)
        c2 = word_to_concept(w2)
        r = str(row['r'])
        if r.find('#') != -1:
            r = r[:r.find('#')]
        rel = word_to_rel(r)
        score = row['score']
        source = row['sources']  # 'http://people.mpi-inf.mpg.de/~ntandon/resources/readme-partwhole.html'
        assoc = [c1, c2, score, source, rel]
        assertions_temp_list.append(assoc)
    web_child_res = pd.DataFrame(assertions_temp_list, index=None)
    web_child_res.to_csv(out_path, sep='\t', index=False, header=False)


def webchild_action_to_conceptnet_format(path_to_tab_sep_file, out_name):
    """
    webchild_action_to_conceptnet_format(path_to_tab_sep_file='wc_action.csv', out_name='wc_spatial_formatted.csv')
    """
    column_names = ['action', 'attribut', 'attribut_value', 'score']
    data = load_df(path_to_tab_sep_file, columns=column_names)
    assertions_temp_list = []
    i = 0
    for index, row in data.iterrows():
        i += 1
        w1 = str(row['action'])
        w2 = str(row['attribut_value'])
        w1_split = w1.split(';')
        w2_split = w2.split(';')
        w1 = ''
        w2 = ''
        for w in w1_split:
            w_split = w.split(' ')
            for wt in w_split:
                inx = wt.find('#')
                if inx != -1:
                    wt = wt[:inx]
                w1 = w1 + wt + ' '
        for w in w2_split:
            w_split = w.split(' ')
            for wt in w_split:
                inx = wt.find('#')
                if inx != -1:
                    wt = wt[:inx]
                w2 = w2 + wt + ' '
        c1 = word_to_concept(w1[:-1])
        c2 = word_to_concept(w2[:-1])
        r = row['attribut']
        rel = word_to_rel(r)
        score = row['score']
        source = 'http://people.mpi-inf.mpg.de/~ntandon/resources/readme-activity.html'
        assoc = [c1, c2, score, source, rel]
        assertions_temp_list.append(assoc)
    web_child_res = pd.DataFrame(assertions_temp_list, index=None)
    web_child_res.to_csv(out_name, sep='\t', index=False, header=False)


def webchild_spatial_to_conceptnet_format(path_to_tab_sep_file, out_name):
    """
    webchild_spatial_to_conceptnet_format(path_to_tab_sep_file='wc_spatial.csv', out_name='wc_spatial_formatted.csv')
    """
    column_names = ['word1', 'locationword', 'artikels_with_counts', 'score']
    data = load_df(path_to_tab_sep_file, columns=column_names)
    assertions_temp_list = []
    i = 0
    for index, row in data.iterrows():
        i += 1
        w1 = str(row['word1'])
        w2 = str(row['locationword'])
        if w1.find('#') != -1:
            w1 = w1[:w1.find('#')]
        if w2.find('#') != -1:
            w2 = w2[:w2.find('#')]
        c1 = word_to_concept(w1)
        c2 = word_to_concept(w2)
        r = row['artikels_with_counts']
        rel_value_list = r.split(',')
        rel_value_dict = {}
        for rel_val in rel_value_list:
            rv_split = rel_val.split(' :')
            rel_value_dict[float(rv_split[1])] = rv_split[0]
        maxim = max(rel_value_dict)
        r = 'is located ' + rel_value_dict[maxim]
        rel = word_to_rel(r)
        score = row['score']
        source = 'spatial'
        assoc = [c1, c2, score, source, rel]
        assertions_temp_list.append(assoc)
    web_child_res = pd.DataFrame(assertions_temp_list, index=None)
    web_child_res.to_csv(out_name, sep='\t', index=False, header=False)


"""
Preprocessing
"""


def reduce_webchild(graph_df, out_path=None):
    """
    Take only maximum scoring edge for relation-subgraphs that contain free relation weights
    map all 'comparative'-Relations to /r/comparative
    reduce_webchild(graph_df=wc_cnformat, out_path='wc_cnformat_reduced.csv')
    """
    print('Shape full: ', graph_df.shape)
    filenames_to_reduce = ['WebChild/ConceptNet_Format/webchild_comparative_concepntnetFormat.csv',
                           'WebChild/ConceptNet_Format/webchild_spatial_concepntnetFormat.csv',
                           'WebChild/ConceptNet_Format/webchild_property_concepntnetFormat.csv']
    for filename in filenames_to_reduce:
        reduce_index = graph_df['filename'] == filename
        reduce_df = graph_df[reduce_index]
        reduce_df = reduce_df.sort_values(['score'], ascending=False)  # sort to keep max score only
        reduce_df = reduce_df.drop_duplicates(['word1', 'word2'], keep='first')  # only keeps max score
        graph_df = graph_df[~reduce_index]  # drop all comparative relations
        graph_df = graph_df.append(reduce_df, ignore_index=True)
        graph_df = graph_df.reset_index(drop=True)
        print('Shape after {}: '.format(filename), graph_df.shape)
    print('Shape final: ', graph_df.shape)
    if out_path:
        graph_df.to_csv(out_path, sep='\t', index=False, header=False)
    return graph_df


"""
Generate sentences from WebChild
"""


def relation_to_word(relation):
    """convert a ConceptNet relation to the word it describes"""
    sp = str.split(relation, '/')
    if len(sp) < 2:
        if relation == 'pseudo_root':
            return 'pseudo_root'
        return ""
    else:
        return sp[2]


def webchild_rel_to_sentence(wd1, wd2, rel, _, file, logging=True):
    if logging:
        missed_rel_file = open("./output/sentences/missed_relation_types.txt", "w+")
    w1 = concept_to_word(wd1).replace('_', ' ')
    w2 = concept_to_word(wd2).replace('_', ' ')
    if rel in webchild_relation_type_to_sentence.keys():  # fixed rel, replace according to dict & set article
        if rel in ['/r/agent', '/r/participant', '/r/next', '/r/hasMember', '/r/hasPart']:  # flip words in sentence
            w_t = w2
            w2 = w1
            w1 = w_t
        sentence = webchild_relation_type_to_sentence[rel].format(w1, w2)
    else:  # sentence composed with rel
        if 'property' in file:  # adds relation to sentence
            if rel == '/r/is':
                sentence = 'a {} is {}'.format(w1, w2)  # special case for 'is' relation
            else:
                sentence = 'a {} {} is {}'.format(w1, relation_to_word(rel).replace('_', ' '), w2)
        elif 'spatial' in file:
            sentence = 'a {} {} {}'.format(w1, relation_to_word(rel).replace('_', ' '), w2)
        elif 'comparative' in file:
            sentence = 'a {} {} a {}'.format(w1, relation_to_word(rel).replace('_', ' '), w2)
        else:  # unknown rel
            sentence = None
            if logging:
                missed_rel_file.write(rel)
    return sentence


def webchild_get_all_sentences(graph_df, indices, out_txt, out_csv):
    """
    WebChild cnformat relations to sentences
    webchild_get_all_sentences(graph_df=wc_cnformat, indices=wc_cnformat.index.values, out_txt='wc_sentences.txt', out_csv='wc_sentences.csv')
    """
    text_file = open(out_txt, "w+")
    inx_sentence_list = []
    s_count = 0
    for inx in indices:
        row = graph_df.loc[inx]
        w1 = row['word1']
        w2 = row['word2']
        rel = row['relation']
        file = row['file']
        rel_sentence = webchild_rel_to_sentence(w1, w2, rel, None, file)
        if rel_sentence:
            text_file.write(rel_sentence + '\n')
            s_count += 1
            inx_sentence_list.append([inx, rel_sentence])
    sent_df = pd.DataFrame.from_dict(inx_sentence_list)
    sent_df.to_csv(out_csv, sep='\t')


if __name__ == '__main__':
    # first load the specially formatted files
    webchild_action_to_conceptnet_format(path_to_tab_sep_file='./data/examples/wc_action.csv',
                                         out_name='./output/kgs/wc_action_formatted.csv')
    webchild_spatial_to_conceptnet_format(path_to_tab_sep_file='./data/examples/wc_spatial.csv',
                                          out_name='./output/kgs/wc_spatial_formatted.csv')

    # then load all other subgraphs of webchild like:
    property_df = load_df('./data/examples/wc_property.csv',
                          columns=['x_disambi', 'attr', 'y_disambi', 'x', 'y', 'freq', 'numsources',
                                   'numpatterns', 'source', 'score', 'higher_attr'])
    webchild_to_conceptnet_format(graph_df=property_df, out_path='wc_property_cnformat.csv')

    # merge the graphs after conversion
    concatenate_files(files_to_merge=['wc_property_cnformat.csv', 'wc_action_cnformat.csv', 'wc_spatial_cnformat.csv'],
                      out_path='./output/kgs/wc_cnformat.csv')

    wc_cnformat = load_df('./output/kgs/wc_cnformat.csv',
                          columns=['word1', 'word2', 'weight', 'source', 'relation', 'file'])
    webchild_get_all_sentences(graph_df=wc_cnformat, indices=wc_cnformat.index.values,
                               out_txt='./output/sentences/wc_sentences.txt',
                               out_csv='./output/sentences/wc_sentences.csv')
