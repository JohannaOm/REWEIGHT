
from sentence_construction.graph_to_sentence import get_all_sentences
from util import load_df, word_to_concept, word_to_rel

"""
Get other graphs to same format as ConceptNet
Example: YAGO Taxonomy Subgraph


Put YAGO into ConceptNet format

Changes for YAGO:
-Words need to be mapped to concepts '/c/en/...'
-YAGO Taxonomy only contains 'SubclassOf' relations, to be mapped to 'IsA' relations

After conversion, applying sentence-generation like with ConceptNet (graph_to_sentence.get_all_sentences)
"""


def get_yago_word(word):
    """column_names 0=subject, 1=object, 2=relation"""
    if word.startswith('<'):
        word = word[1:-1]
        word = word.split('_')
        if word[0] == 'wikicat':
            word = ' '.join(w for w in word[1:])
        elif word[0] == 'wordnet':
            word = ' '.join(w for w in word[1:-1])
        else:
            return
        return word
    else:
        return


def get_yago_source(concept):
    concept = concept[1:-1]
    concept_split = concept.split('_')
    return concept_split[0]


def yago_taxonomy_to_conceptnet_format(yago_path, out_file):
    """
    yago_taxonomy_to_conceptnet_format(yago_path='yago_taxonomy.tsv', out_file='yago_cnformat.csv')
    """
    # columns names [id, word1, relation, word2]
    graph = open(yago_path, "r")
    result_file = open(out_file, "w+")
    counter = 0
    errors = 0
    for line in graph:
        if counter == 0:
            counter += 1
            continue
        counter += 1
        if counter % 10000 == 0:
            print('Counter: ', counter)
        line_split = line.split('\t')
        relation = line_split[2].split(':')[1]
        word1 = get_yago_word(line_split[1])
        word2 = get_yago_word(line_split[3])
        if word1 and word2:
            word1 = word_to_concept(word1)
            word2 = word_to_concept(word2)
            source = '' + get_yago_source(line_split[1]) + ';' + get_yago_source(line_split[3])
            if relation == 'subClassOf':
                relation = word_to_rel(relation)
                line_to_write = word1 + '\t' + word2 + '\t' + '1' + '\t' + source + '\t' + relation + '\n'
                result_file.write(line_to_write)
            else:
                print("Wrong Relation Format: ", line)
                errors += 1
        else:
            print(counter, ' : Wrong Format of words : ', line)
            errors += 1
    print("Total facts transformed to CN Format: ", counter)
    print('Total Errors : ', errors)


if __name__ == '__main__':
    yago_taxonomy_to_conceptnet_format(yago_path='./data/examples/yago_taxonomy.tsv',
                                       out_file='./output/kgs/yago_cnformat.csv')
    yago_df = load_df('./output/kgs/yago_cnformat.csv', columns=['word1', 'word2', 'weight', 'source', 'relation'])
    get_all_sentences(graph_df=yago_df, indices=yago_df.index.values,
                      out_txt='./output/sentences/yago_sentences.txt', out_csv='./output/sentences/yago_sentences.csv')
