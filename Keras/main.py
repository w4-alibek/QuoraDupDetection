#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Trains LSTM model with train.csv. How to run:
python main.py

<REQUIRED FLAGS TO RUN>
--raw_train_data ~chungshik/quora_data/data/train.csv
--raw_train_nlp_features ~alibek/QuoraDupDetection/nlp_features_train.csv
--raw_train_non_nlp_features ~alibek/QuoraDupDetection/non_nlp_features_train.csv
--raw_test_data ~chungshik/quora_data/data/train.csv
--raw_test_nlp_features ~alibek/QuoraDupDetection/nlp_features_test.csv
--raw_test_non_nlp_features ~alibek/QuoraDupDetection/non_nlp_features_test.csv
--word_embedding_path ~chungshik/quora_data/word_embeddings/glove.840B.300d.txt
--embedding_vector_dimension 300
--path_save_best_model ~alibek/QuoraDupDetection/Keras/models

<OPTIONAL FLAGS TO RUN>
--generate_csv_submission_best_model True
--early_stopping_patience 50
--learning_rate 0.0005

<USE THIS FLAG ONLY WHEN GENERATING SUBMISSION FILE FROM SAVED MODEL.>
--model_file_to_load
"""
from __future__ import print_function

from keras import optimizers
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.core import Lambda
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras.layers.merge import multiply
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from time import gmtime, strftime
import numpy
import os
import pandas as pd
import tensorflow as tf

import glove_embedding as embedding
from features import nlp

# Word embeddings
tf.flags.DEFINE_string("word_embedding_path", '', "Where the word embedding vectors are located.")
tf.flags.DEFINE_integer("embedding_vector_dimension", None, "Word embedding vector's dimension.")

# Training
tf.flags.DEFINE_bool("remove_stopwords", True, "Remove stop words")
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.flags.DEFINE_float("validation_split", 0.2, "Split train.csv file into train and validation")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size")
tf.flags.DEFINE_integer("max_sequence_length", 100, "Maximum length of question length")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.flags.DEFINE_integer("train_extra_num_epoch", 0, "Train the model full data set.")
tf.flags.DEFINE_integer("early_stopping_patience", 5,
                        "Number of epochs with no improvement after which training will be stopped")
tf.flags.DEFINE_string("path_save_best_model", None, "Path to save best model of training")
tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_train_nlp_features", None, "Where the raw train nlp features is stored")
tf.flags.DEFINE_string("raw_train_non_nlp_features", None,
                       "Where the raw train non nlp features is stored")
tf.flags.DEFINE_string("optimizer", "nadam",
                       "Optimization method. One of 'adadelta', 'adam', nadam"
                       "'adamax', 'sgd', 'adagrad', 'rmsprop'")


# Testing
tf.flags.DEFINE_string("raw_test_data", None, "Where the raw test data is stored.")
tf.flags.DEFINE_string("raw_test_nlp_features", None, "Where the raw test nlp features is stored.")
tf.flags.DEFINE_string("raw_test_non_nlp_features", None,
                       "Where the raw test non nlp features is stored.")
tf.flags.DEFINE_bool("generate_csv_submission_best_model", False,
                     "Generate csv submission file base on last model.")

# LSTM model
tf.flags.DEFINE_integer("lstm_out_dimension", 50,
                        "Hidden state dimension (LSTM output vector dimension)")
tf.flags.DEFINE_float("zoneout", 0.2, "Apply zoneout (dropout) to F gate")

# Model
tf.flags.DEFINE_string("model_file_to_load", None, "Where the model weights file is located")
tf.flags.DEFINE_string("model", "base_model",
                       "Name of a model to run. One of 'base_model', 'bidirectional_rnn', 'qrnn'.")

FLAGS = tf.flags.FLAGS
NOW_DATETIME = strftime("%Y-%m-%d-%H-%M-%S", gmtime())


def build_lstm_layer():
    if FLAGS.model == "base_model":
        lstm_layer = LSTM(FLAGS.lstm_out_dimension, recurrent_dropout=FLAGS.zoneout)
    elif FLAGS.model == "bidirectional_rnn":
        lstm_layer = Bidirectional(
            LSTM(FLAGS.lstm_out_dimension, recurrent_dropout=FLAGS.zoneout),
            merge_mode='concat')
    else:
        lstm_layer = LSTM(FLAGS.lstm_out_dimension, recurrent_dropout=FLAGS.zoneout)
    return lstm_layer


def generate_padded_sequence(question1_list, question2_list, tokenizer):
    sequences1 = tokenizer.texts_to_sequences(question1_list)
    sequences2 = tokenizer.texts_to_sequences(question2_list)

    data_1 = pad_sequences(sequences1, maxlen=FLAGS.max_sequence_length)
    data_2 = pad_sequences(sequences2, maxlen=FLAGS.max_sequence_length)

    return data_1, data_2


def build_model(lstm_layer_lhs, lstm_layer_rhs, input_sequence_1, input_sequence_2, features_input):

    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)

    # Square difference
    addition = add([lstm_layer_lhs, lstm_layer_rhs])
    minus_lstm_layer_rhs = Lambda(lambda x: -x)(lstm_layer_rhs)
    merged = add([lstm_layer_lhs, minus_lstm_layer_rhs])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(150, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_sequence_1, input_sequence_2, features_input], outputs=out)

    if FLAGS.optimizer == "adam":
        optimizer = optimizers.Adam(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "adadelta":
        optimizer = optimizers.Adadelta(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "adagrad":
        optimizer = optimizers.Adagrad(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "adamax":
        # It is a variant of Adam based on the infinity norm.
        # Default parameters follow those provided in the paper.
        # Default lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0
        optimizer = optimizers.Adamax(lr=FLAGS.learning_rate)
    elif FLAGS.optimizer == "nadam":
        # Much like Adam is essentially RMSprop with momentum,
        # Nadam is Adam RMSprop with Nesterov momentum.
        # Default same as Adamax with ..., schedule_decay=0.004
        optimizer = optimizers.Nadam(lr=FLAGS.learning_rate)
    else:
        optimizer = optimizers.Nadam(lr=FLAGS.learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return model


def train(model, train_set):
    csv_logger = CSVLogger('./tmp/' + NOW_DATETIME + '_training.log')
    logging = TensorBoard(log_dir='./logs',
                        histogram_freq=0,
                        batch_size=FLAGS.batch_size,
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None)
    best_model_path = os.path.join(FLAGS.path_save_best_model, NOW_DATETIME + "_best_model.h5")
    early_stopping = EarlyStopping(monitor="val_loss", patience=FLAGS.early_stopping_patience)
    model_checkpoint = ModelCheckpoint(best_model_path,
                                       save_best_only=True,
                                       save_weights_only=True)

    print("Path to best model from training: " + best_model_path)

    model.fit([train_set[0], train_set[1], train_set[2]],
              train_set[3],
              validation_split=FLAGS.validation_split,
              epochs=FLAGS.num_epochs,
              batch_size=FLAGS.batch_size,
              shuffle=True,
              callbacks=[early_stopping, model_checkpoint, logging, csv_logger],
              verbose=1)
    print("'Best model from training saved: " + NOW_DATETIME + "_best_model.h5")


def generate_csv_submission(model, test_set, model_type):
        # Testing and generating submission csv
        print("Testing model...")
        preds = model.predict([test_set[0], test_set[1], test_set[2]],
                              batch_size=FLAGS.batch_size,
                              verbose=1)
        print("Generating preds_"+ NOW_DATETIME + ".csv ...")
        submission = pd.DataFrame({"is_duplicate": preds.ravel(), "test_id": test_set[3]})
        submission.to_csv("predictions/preds_"+ NOW_DATETIME + model_type + ".csv", index=False)


def main():
    embedding_layer, train_labels, question1_list, question2_list, tokenizer = embedding\
        .process_data(FLAGS.word_embedding_path,
                      FLAGS.raw_train_data,
                      FLAGS.embedding_vector_dimension,
                      FLAGS.max_sequence_length)

    # Load nlp features for train data set.
    print("Reading nlp features...")
    train_nlp_features = pd.read_csv(FLAGS.raw_train_nlp_features)
    train_non_nlp_features = pd.read_csv(FLAGS.raw_train_non_nlp_features);
    train_features = numpy.hstack((train_nlp_features, train_non_nlp_features))

    lstm_layer = build_lstm_layer()

    # Specifying model input shape
    input_sequence_1 = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
    embedded_sequences_1 = embedding_layer(input_sequence_1)

    input_sequence_2 = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
    embedded_sequences_2 = embedding_layer(input_sequence_2)

    features_input = Input(shape=(train_features.shape[1],), dtype="float32")

    # Feeding embedded sequence to LSTM layers
    lstm_layer_lhs = lstm_layer(embedded_sequences_1)
    lstm_layer_rhs = lstm_layer(embedded_sequences_2)

    # Build a model
    model = build_model(lstm_layer_lhs,
                        lstm_layer_rhs,
                        input_sequence_1,
                        input_sequence_2,
                        features_input)

    # Padding sequence
    train_data_1, train_data_2 = generate_padded_sequence(
        question1_list,
        question2_list,
        tokenizer)

    # Train a model if model_file_to_load flag not specified. "GENERATING MODEL"
    if FLAGS.model_file_to_load is None:
        train_set = [train_data_1,train_data_2, train_features, train_labels]
        train(model, train_set)
    else:
        model.load_weights(FLAGS.model_file_to_load)

    print("Read test.csv file...")
    # Read test data and do same for test data.
    test = pd.read_csv(FLAGS.raw_test_data)
    test["question1"] = test["question1"].fillna("").apply(nlp.clean_text)\
        .apply(nlp.remove_stop_words_and_punctuation).apply(nlp.word_net_lemmatize)
    test["question2"] = test["question2"].fillna("").apply(nlp.clean_text)\
        .apply(nlp.remove_stop_words_and_punctuation).apply(nlp.word_net_lemmatize)

    test_nlp_features = pd.read_csv(FLAGS.raw_test_nlp_features)
    test_non_nlp_features = pd.read_csv(FLAGS.raw_test_non_nlp_features)
    test_features = numpy.hstack((test_nlp_features, test_non_nlp_features))

    test_data_1, test_data_2 = generate_padded_sequence(test["question1"],
                                                        test["question2"],
                                                        tokenizer)
    test_id = test["test_id"]
    test_set = [test_data_1, test_data_2, test_features, test_id]

    # Generate csv file for submission with best model
    if FLAGS.generate_csv_submission_best_model:
        best_model_path = os.path.join(FLAGS.path_save_best_model, NOW_DATETIME + "_best_model.h5")
        model.load_weights(best_model_path)
        generate_csv_submission(model, test_set, "")


if __name__ == "__main__":
    main()
