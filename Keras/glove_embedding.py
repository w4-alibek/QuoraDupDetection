from __future__ import print_function

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

import read_data
import features

def process_data(glove_embedding_path, raw_train_data, embedding_dim, max_sequence_length):
    print("Reading GloVe embedding...")
    embeddings_index = read_data.read_GloVe(glove_embedding_path)
    print("Reading train data set...")
    train = pd.read_csv(raw_train_data)
    print("Text processing...")
    train["question1"] = train["question1"].fillna("").apply(features.clean_text) \
        .apply(features.remove_stop_words_and_punctuation).apply(features.word_net_lemmatize)
    train["question2"] = train["question2"].fillna("").apply(features.clean_text)\
        .apply(features.remove_stop_words_and_punctuation).apply(features.word_net_lemmatize)

    labels = np.array(train["is_duplicate"])  # list of label ids
    question1_list = np.array(train["question1"])
    question2_list = np.array(train["question2"])


    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(np.append(question1_list, question2_list))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words =len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # count_none = 0
    # back_of_words = []
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # if embedding_vector is None:
        #     count_none = count_none + 1
        #     back_of_words.append(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # print('# not found words', count_none)
    # not_found_words = pd.DataFrame({"word": back_of_words})
    # not_found_words.to_csv("./with_norm.csv", index=False, encoding="utf-8")
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)


    return embedding_layer, labels, question1_list, question2_list, tokenizer
