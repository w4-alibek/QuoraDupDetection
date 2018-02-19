import numpy as np

# GLOVE_DIR = '/Users/skelter/Desktop/QuoraDupDetection/Keras/glove.6B.50d.txt'

def read_GloVe(GLOVE_DIR):
    embeddings_index = {}
    with open(GLOVE_DIR) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    f.close()
    
    return embeddings_index;