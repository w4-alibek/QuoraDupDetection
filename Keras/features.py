#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
refs: https://github.com/aerdem4/kaggle-quora-dup
"""
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import string
import tensorflow as tf
import util
import os, sys

STOP_WORDS = stopwords.words("english")

tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_test_data", None, "Where the raw train data is stored.")

FLAGS = tf.flags.FLAGS


def lemmatize_word(word):
    if len(word) < 4:
        return word
    return WordNetLemmatizer().lemmatize(WordNetLemmatizer().lemmatize(word, "n"), "v")


def word_net_lemmatize(text):
    """Normalize the given list of words. Return list of normalized word
    """
    return ' '.join([lemmatize_word(w) for w in text.split()])


def clean_text(text):

    r = ((u",000,000", u"m"),
         (u",000", u"k"),
         (u"′", u"'"),
         (u"’", u"'"),
         (u"won't", u"will not"),
         (u"cannot", u"can not"),
         (u"can't", u"can not"),
         (u"n't", u" not"),
         (u"what's", u"what is"),
         (u"it's", u"it is"),
         (u"'ve", u" have"),
         (u"i'm", u"i am"),
         (u"'re", u" are"),
         (u"he's", u"he is"),
         (u"she's", u"she is"),
         (u"'s", u" own"),
         (u"%", u" percent "),
         (u"₹", u" rupee "),
         (u"$", u" dollar "),
         (u"€", u" euro "),
         (u"'ll", u" will"),
         (u"=", u" equal "),
         (u"+", u" plus "))

    text = text.lower()
    for original, replaced in r:
        try:
            text = text.replace(original, replaced)
        except UnicodeDecodeError:
            print repr(text), repr(original), repr(replaced)

    text = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', text)
    text = re.sub(r"([0-9]+)000000", r"\1m", text)
    text = re.sub(r"([0-9]+)000", r"\1k", text)
    return text


def remove_stop_words_and_punctuation(text):
    """Given text as string. Removes stop words from text and
    return list of words without stop word
    """
    stop_words = stopwords.words('english')
    stop_words.extend(list(string.punctuation))
    stop_words = set(stop_words)
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_words)

# Generate nlp features for test and train data set.
def get_token_features(question1, question2):
    token_features = [0.0]*10

    question1_tokens = question1.split()
    question2_tokens = question2.split()

    if len(question1_tokens) == 0 or len(question2_tokens) == 0:
        return token_features

    question1_words = set([word for word in question1_tokens if word not in STOP_WORDS])
    question2_words = set([word for word in question2_tokens if word not in STOP_WORDS])

    question1_stops = set([word for word in question1_tokens if word in STOP_WORDS])
    question2_stops = set([word for word in question2_tokens if word in STOP_WORDS])

    common_word_count = len(question1_words.intersection(question2_words))
    common_stop_count = len(question1_stops.intersection(question2_stops))
    common_token_count = len(set(question1_tokens).intersection(set(question2_tokens)))

    token_features[0] = common_word_count / (min(len(question1_words),
                                                 len(question2_words)) + 0.0001)
    token_features[1] = common_word_count / (max(len(question1_words),
                                                 len(question2_words)) + 0.0001)
    token_features[2] = common_stop_count / (min(len(question1_stops),
                                                 len(question2_stops)) + 0.0001)
    token_features[3] = common_stop_count / (max(len(question1_stops),
                                                 len(question2_stops)) + 0.0001)
    token_features[4] = common_token_count / (min(len(question1_tokens),
                                                  len(question2_tokens)) + 0.0001)
    token_features[5] = common_token_count / (max(len(question1_tokens),
                                                  len(question2_tokens)) + 0.0001)
    token_features[6] = int(question1_tokens[-1] == question2_tokens[-1])
    token_features[7] = int(question1_tokens[0] == question2_tokens[0])
    token_features[8] = abs(len(question1_tokens) - len(question2_tokens))
    token_features[9] = (len(question1_tokens) + len(question2_tokens))/2
    return token_features



def get_longest_substr_ratio(question1, question2):
    strs = list(util.lcsubstrings(question1, question2))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(question1), len(question2)) + 1)


def extract_features(nlp_features):
    nlp_features["question1"] = nlp_features["question1"].fillna("").apply(clean_text)
    nlp_features["question2"] = nlp_features["question2"].fillna("").apply(clean_text)

    print("token features...")
    token_features = nlp_features.apply(
        lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    nlp_features["cwc_min"] = list(map(lambda x: x[0], token_features))
    nlp_features["cwc_max"] = list(map(lambda x: x[1], token_features))
    nlp_features["csc_min"] = list(map(lambda x: x[2], token_features))
    nlp_features["csc_max"] = list(map(lambda x: x[3], token_features))
    nlp_features["ctc_min"] = list(map(lambda x: x[4], token_features))
    nlp_features["ctc_max"] = list(map(lambda x: x[5], token_features))
    nlp_features["last_word_eq"] = list(map(lambda x: x[6], token_features))
    nlp_features["first_word_eq"] = list(map(lambda x: x[7], token_features))
    nlp_features["abs_len_diff"] = list(map(lambda x: x[8], token_features))
    nlp_features["mean_len"] = list(map(lambda x: x[9], token_features))

    print("fuzzy features..")
    nlp_features["token_set_ratio"] = nlp_features.apply(
        lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    nlp_features["token_sort_ratio"] = nlp_features.apply(
        lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    nlp_features["fuzz_ratio"] = nlp_features.apply(
        lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    nlp_features["fuzz_partial_ratio"] = nlp_features.apply(
        lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    nlp_features["longest_substr_ratio"] = nlp_features.apply(
        lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return nlp_features

# print("Extracting features for train...")
# train_nlp_features = pd.read_csv(FLAGS.raw_train_data)
# train_nlp_features = extract_features(train_nlp_features)
# train_nlp_features.drop(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"],
#                         axis=1,
#                         inplace=True)
# train_nlp_features.to_csv("data/nlp_features_train.csv", index=False)
#
# print("Extracting features for test...")
# test_nlp_features = pd.read_csv(FLAGS.raw_test_data)
# test_nlp_features = extract_features(test_nlp_features)
# test_nlp_features.drop(["test_id", "question1", "question2"], axis=1, inplace=True)
# test_nlp_features.to_csv("data/nlp_features_test.csv", index=False)
