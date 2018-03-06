#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from features import nlp

# Set these flags to generate feature CSV files.
tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data are to be stored.")
tf.flags.DEFINE_string("raw_test_data", None, "Where the raw test data are to be stored.")
tf.flags.DEFINE_string("raw_preprocessed_train_data", None,
                       "Where the raw preprocessed train data are to be stored.")
tf.flags.DEFINE_string("raw_preprocessed_test_data", None,
                       "Where the raw preprocessed test data are to be stored.")
tf.flags.DEFINE_string("pos_tagger", None, "Where the pos tagger map located.")
tf.flags.DEFINE_string("pos_tag_map", None, "Where the pos tag map data located.")
tf.flags.DEFINE_string("word_frequency", None, "Where the word frequency data located.")

# --raw_train_data=/Users/skelter/Desktop/QuoraDupDetection/Keras/train_data/train.500.csv
# --raw_test_data=/Users/skelter/Desktop/QuoraDupDetection/Keras/train_data/test.500.csv
# --pos_tagger=/Users/skelter/Desktop/QuoraDupDetection/Keras/data/pos_tagger_for_top_5000_words.csv
# --word_frequency=/Users/skelter/Desktop/QuoraDupDetection/Keras/data/top_5000_words.csv
# --pos_tag_map=/Users/skelter/Desktop/QuoraDupDetection/Keras/data/pos_tag_map.csv

FLAGS = tf.flags.FLAGS


def remove_rare_words(text, top_words):
    list_word = []
    text = text.split()
    for word in text:
        if word in top_words:
            list_word.append(word)
        else:
            # Randomly choosen word(5032th frequently used word in train set).
            list_word.append("alliance")
    return ' '.join(list_word)


def top_7_words(word_freq, word_list):
    sorted_by_freq = sorted([(word, word_freq[word]) for word in word_list],
                            key=lambda tup: tup[1],
                            reverse=True)
    if len(sorted_by_freq) > 7:
        sorted_by_freq = sorted_by_freq[:7]
    return [word_count[0] for word_count in sorted_by_freq]


def _subliner_term_frequency(word, tokenized_words):
    count = tokenized_words.count(word)
    if count == 0:
        return 0
    return 1 + math.log(count)


def _inverse_document_frequencies(tokenized_sentences, set_of_words):
    idf_values = {}
    # set_of_words = set([word for sentence in tokenized_sentences for word in sentence])
    for word in set_of_words:
        contains_token = 0.00001
        for sentence in tokenized_sentences:
            contains_token += word in sentence
        idf_values[word] = 1 + math.log(len(tokenized_sentences)/contains_token)
    return idf_values


def generate_tfxidf_feature(train, word_freq, set_of_words, category):
    print("Generating TFxIDF feature "+category+"...")
    tokenized_sentences = np.hstack((train["question1"], train["question2"]))
    tokenized_sentences = [tokenized_sentence.split() for tokenized_sentence in tokenized_sentences]
    inverse_document_frequencies = _inverse_document_frequencies(tokenized_sentences, set_of_words)

    feature = []
    feature_col = ["tfidf1", "tfidf2", "tfidf3", "tfidf4", "tfidf5", "tfidf6", "tfidf7",
                   "tfidf8", "tfidf9", "tfidf10", "tfidf11", "tfidf12", "tfidf13", "tfidf14"]

    for question1, question2 in np.stack((train["question1"], train["question2"]), axis=-1):
        tokenized_sentence_1 = question1.split()
        tokenized_sentence_2 = question1.split()
        # Get top 7 words from question1 and question2
        top_7_words_1 = top_7_words(word_freq, tokenized_sentence_1)
        top_7_words_2 = top_7_words(word_freq, tokenized_sentence_2)

        feature_weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for index in range(len(top_7_words_1)):
            word = top_7_words_2[index]
            subliner_term_frequency = _subliner_term_frequency(word, tokenized_sentence_1)
            feature_weight[index] = (subliner_term_frequency * inverse_document_frequencies[word])

        for index in range(len(top_7_words_2)):
            word = top_7_words_2[index]
            subliner_term_frequency = _subliner_term_frequency(word, tokenized_sentence_1)
            feature_weight[7+index] = (subliner_term_frequency * inverse_document_frequencies[word])
        feature.append(feature_weight)

        if len(feature) % 50000 == 0:
            print("Step: " + len(feature))

    save_dict = pd.DataFrame()
    for index in range(len(feature_col)):
        save_dict[feature_col[index]] = [column[index] for column in feature]

    save_dict.to_csv("./tfidf_feature_"+category+".csv", index=False)


def generate_pos_tag_feature(train, pos_tag_map, pos_tagger, word_freq, category):

    print("Generating pos tag feature "+category+"...")
    feature = []
    feature_col = ["pos1", "pos2", "pos3", "pos4", "pos5", "pos6", "pos7",
                   "pos8", "pos9", "pos10", "pos11", "pos12", "pos13", "pos14"]

    list_of_top_7 = []

    train["question1"] = train["question1"].fillna("")
    train["question2"] = train["question2"].fillna("")

    for question1, question2 in np.stack((train["question1"], train["question2"]), axis=-1):
        # Get top 7 words from question1 and question2
        top_7_words_1 = top_7_words(word_freq, question1.split())
        top_7_words_2 = top_7_words(word_freq, question2.split())

        list_of_top_7.append((top_7_words_1, top_7_words_2))

        pos_tag_1 = [pos_tagger[str(word)] for word in top_7_words_1]
        pos_tag_2 = [pos_tagger[str(word)] for word in top_7_words_2]

        feature_weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for index in range(len(pos_tag_1)):
            feature_weight[index] = pos_tag_map[pos_tag_1[index]] / 25.0000001
        for index in range(len(pos_tag_2)):
            feature_weight[7 + index] = pos_tag_map[pos_tag_2[index]] / 25.0000001
        feature.append(feature_weight)

    save_dict = pd.DataFrame()
    for index in range(len(feature_col)):
        save_dict[feature_col[index]] = [column[index] for column in feature]

    save_dict.to_csv("./pos_tag_feature_"+category+".csv", index=False)


def preprocess_data(set_of_words):
    print("Reading train.csv...")
    train = pd.read_csv(FLAGS.raw_train_data)
    print("Cleaning train quesitons...")
    train["question1"] = train["question1"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)
    train["question2"] = train["question2"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)

    print("Reading test.csv...")
    test = pd.read_csv(FLAGS.raw_test_data)
    print("Cleaning test quesitons...")
    test["question1"] = test["question1"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)
    test["question2"] = test["question2"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)

    print("Removing rare words from train.csv...")
    # Remove rare words for data set.
    train["question1"] = train["question1"].apply(lambda x: remove_rare_words(x, set_of_words))
    train["question2"] = train["question2"].apply(lambda x: remove_rare_words(x, set_of_words))

    print("Removing rare words from test.csv...")
    test["question1"] = test["question1"].apply(lambda x: remove_rare_words(x, set_of_words))
    test["question2"] = test["question2"].apply(lambda x: remove_rare_words(x, set_of_words))

    print("Saving processed data...")
    # Save preprocessed data.
    train.to_csv("./preprocessed_train.csv", index=False)
    test.to_csv("./preprocessed_test.csv", index=False)

    return train, test


def read_preprocessed_data():
    print("Reading preprocessed train.csv...")
    train = pd.read_csv(FLAGS.raw_preprocessed_train_data)
    train["question1"] = train["question1"].fillna("")
    train["question2"] = train["question2"].fillna("")

    print("Reading preprocessed test.csv...")
    test = pd.read_csv(FLAGS.raw_preprocessed_test_data)
    test["question1"] = test["question1"].fillna("")
    test["question2"] = test["question2"].fillna("")

    return train, test


print("Reading pos_tagger_for_top_5000.csv...")
_pos_tagger = pd.read_csv(FLAGS.pos_tagger)
# Generate pos tagger map.
pos_data = [tag for tag in _pos_tagger["pos_tag"]]
pos_tagger = pd.Series(data=pos_data, index=_pos_tagger["word"]).to_dict()

print("Reading word_freq.csv...")
word_freq = pd.read_csv(FLAGS.word_frequency)

print("Reading pos_tag_map.csv...")
pos_tag_map = pd.read_csv(FLAGS.pos_tag_map)
pos_data = [tag for tag in pos_tag_map["id"]]
pos_tag_map = pd.Series(pos_data, index=pos_tag_map["pos_tag"]).to_dict()

# Generate set of words.
set_of_words = [unicode(word, errors='ignore') for word in word_freq["word"]]

# Generate word frequency map.
word_freq = pd.Series(word_freq["freq"], index=word_freq["word"]).to_dict()

if FLAGS.raw_preprocessed_train_data and FLAGS.raw_preprocessed_test_data:
    train, test = read_preprocessed_data()
else:
    train, test = preprocess_data(set_of_words)

# Generate features
# generate_pos_tag_feature(train, pos_tag_map, pos_tagger, word_freq, "train")
# generate_pos_tag_feature(test, pos_tag_map, pos_tagger, word_freq, "test")
#
# generate_tfxidf_feature(train, word_freq, set_of_words, "train")
generate_tfxidf_feature(test, word_freq, set_of_words, "test")