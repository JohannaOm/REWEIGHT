import pandas as pd

from util import load_df, get_en_idx

relation_types_to_sentence = {'/r/RelatedTo': 'is related to', '/r/FormOf': 'is a form of',
                              '/r/IsA': 'is', '/r/PartOf': 'is a part of', '/r/HasA': 'is a part of',
                              '/r/UsedFor': 'is used for', '/r/CapableOf': 'is able to',
                              '/r/AtLocation': 'is located at', '/r/Causes': 'is caused by',
                              '/r/HasSubevent': 'occurs as part of', '/r/HasFirstSubevent': 'begins with',
                              '/r/HasLastSubevent': 'concludes with', '/r/HasPrerequisite': 'needs',
                              '/r/HasProperty': 'is', '/r/MotivatedByGoal': 'is motivated by',
                              '/r/ObstructedBy': 'can be prevented by', '/r/Desires': 'want',
                              '/r/CreatedBy': 'created by', '/r/Synonym': 'is similar to',
                              '/r/Antonym': 'is different from', '/r/DistinctFrom': 'is distinct from',
                              '/r/DerivedFrom': 'is derived from', '/r/SymbolOf': 'is a symbol of',
                              '/r/DefinedAs': 'is defined as', '/r/MannerOf': 'is a',
                              '/r/LocatedNear': 'is located near',
                              '/r/HasContext': 'has context', '/r/SimilarTo': 'is similar to',
                              '/r/EtymologicallyRelatedTo': 'has a common origin with',
                              '/r/EtymologicallyDerivedFrom': 'is derived from',
                              '/r/CausesDesire': 'makes to want to', '/r/MadeOf': 'is made of',
                              '/r/ReceivesAction': 'can be', '/r/ExternalURL': 'External URL',
                              '/r/NotDesires': 'does not want', '/r/InstanceOf': 'is instance of',
                              '/r/subClassOf': 'is'}

"""
Sentence construction
"""


def concept_to_word(concept):
    """convert a ConceptNet concept to the word it describes"""
    sp = str.split(concept, '/')
    if len(sp) < 3:
        if concept == 'pseudo_root':
            return 'pseudo_root'
        return ""
    else:
        return sp[3]


def rel_to_sentence(wd1, wd2, rel, logging=True):
    if logging:
        missing_rel_file = open("./output/sentences/missed_relation_types.txt", "w+")
    w1 = concept_to_word(wd1).replace('_', ' ')
    w2 = concept_to_word(wd2).replace('_', ' ')
    if rel in relation_types_to_sentence.keys():
        sentence = "a " + w1 + " " + (relation_types_to_sentence[rel]).replace('_', ' ') + " a " + w2
        if rel in ['/r/Causes', '/r/HasA']:
            w_t = w2
            w2 = w1
            w1 = w_t
            sentence = "a " + w1 + " " + (relation_types_to_sentence[rel]).replace('_', ' ') + " a " + w2
        return sentence
    elif logging:
        missing_rel_file.write(rel)
    return None


def get_all_sentences(graph_df, indices, out_txt, out_csv):
    """
    Takes graph_df with columns ['word1', 'word2', 'relation'] as input and constructs sentences for all indices.
    Saves sentences as .txt for BERT inputs and sentence + index as .csv for mapping back new weights to the graph.
    get_all_sentences(graph_df=conceptnet, indices=get_en_idx(conceptnet), out_txt='cn_en_sentences.txt', out_csv='cn_en_sentences.csv')
    """
    text_file = open(out_txt, "w+")
    inx_sentence_list = []
    s_count = 0
    for inx in indices:
        row = graph_df.loc[inx]
        w1 = row['word1']
        w2 = row['word2']
        rel = row['relation']
        rel_sentence = rel_to_sentence(w1, w2, rel)
        if rel_sentence:
            text_file.write(rel_sentence + '\n')
            s_count += 1
            inx_sentence_list.append([inx, rel_sentence])
    sent_df = pd.DataFrame.from_dict(inx_sentence_list)
    sent_df.to_csv(out_csv, sep='\t')


"""
Preprocessing for BERT LMs
"""


def split_text_file(file_path, out_template, chunk_size):
    """
    split large txt file in many small txt files of same size for multiprocessing on LMs
    split_text_file(file_path='sentences.txt', out_template='sentences_chunk{}.txt', chunk_size=200000)
    """
    file_to_split = open(file_path, 'r')
    file_name_inx = 0
    file_to_write = open(out_template.format(file_name_inx), 'w+')
    for i, line in enumerate(file_to_split):
        if (i + 1) % chunk_size == 0:
            print(i)
            file_to_write.close()
            file_name_inx += 1
            file_to_write = open(out_template.format(file_name_inx), 'w+')
        file_to_write.write(line)


def split_long_words(file_path, out_path, max_len):
    """
    Split very long words so BERT doesnt crash
    split_long_words(file_path='sentences_chunk0.txt', out_path='sentences_chunk0_split.txt', max_len=43)
    """
    print("Start Split")
    file_to_split = open(file_path, 'r')
    file_to_write = open(out_path, 'w+')
    for line in file_to_split:
        line_split = line.split(' ')
        sentence_new = ''
        for split in line_split:
            if len(split) > max_len:
                print(split)
                w_split = (split[0 + i:max_len + i] for i in range(0, len(split), max_len))
                sentence_new = sentence_new + (' '.join(word for word in w_split)) + ' '
            else:
                sentence_new = sentence_new + split + ' '

        new_line = sentence_new[:-1]
        file_to_write.write(new_line)
    print("End Split")


if __name__ == '__main__':
    conceptnet = load_df('./data/examples/conceptnet_toy.csv',
                         columns=['word1', 'word2', 'weight', 'source', 'relation', 'file'])
    get_all_sentences(graph_df=conceptnet,
                      indices=get_en_idx(conceptnet),
                      out_txt='./output/sentences/cn_toy_sentences.txt',
                      out_csv='./output/sentences/cn_toy_sentences.csv')

    # if you wish to run multiple language models on chunks of your data at once:
    # split_text_file(file_path='cn_en_sentences.txt', out_template='cn_en_sentences_{}.txt', chunk_size=200000)

    # in case one of your chunks crashes BERT due to length of single words:
    # split_long_words(file_path='./output/sentences/cn_toy_sentences.txt',
    #                  out_path='./output/sentences/cn_toy_sentences_split.txt', max_len=43)
