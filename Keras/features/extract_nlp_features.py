#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import pandas as pd
import tensorflow as tf

from nlp import extract_features

# Set these flags to generate feature CSV files.
tf.flags.DEFINE_string(
    "raw_train_data", None,
    "Where the raw train data are to be stored.")
tf.flags.DEFINE_string(
    "raw_test_data", None,
    "Where the raw test data are to be stored.")
tf.flags.DEFINE_string(
    "path_train_nlp_features", None,
    "Path to save train nlp features.")
tf.flags.DEFINE_string(
    "path_test_nlp_features", None,
    "Path to save test nlp features.")

FLAGS = tf.flags.FLAGS

# Generating feature CSV file for training data.
if FLAGS.raw_train_data is not None:
    print("Extracting features for train...")
    train_nlp_features = pd.read_csv(FLAGS.raw_train_data)
    train_nlp_features = extract_features(train_nlp_features)
    train_nlp_features.drop(
        ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"],
        axis=1, inplace=True)
    train_nlp_features.to_csv(FLAGS.path_train_nlp_features, index=False)

# Generating feature CSV file for test data.
if FLAGS.raw_test_data is not None:
    print("Extracting features for test...")
    test_nlp_features = pd.read_csv(FLAGS.raw_test_data)
    test_nlp_features = extract_features(test_nlp_features)
    test_nlp_features.drop(
        ["test_id", "question1", "question2"], axis=1, inplace=True)
    test_nlp_features.to_csv(FLAGS.path_test_nlp_features, index=False)