"""Build embedding dictionaries from various datasets
wordid -> tf vector
"""

import datasets

def build_glove_embeddings(word_to_id, filename, dimension):
    embeddings = [[0] * dimension for i in range(len(word_to_id))] # is it correct to initialize to tensor 0 if word is not found?
    for line in datasets.glove_dataset(dimension, filename):
        if line[0]:
            embeddings[word_to_id[line[0]]] = map(float, line[1:])

    return embeddings
