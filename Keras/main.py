"""Trains LSTM model with train.csv. How to run:
python main.py
--raw_train_data ~chungshik/quora_data/data/train.csv
--word_embedding_path ~chungshik/quora_data/word_embeddings/glove.840B.300d.txt
--embedding_vector_dimension 300 --batch_size 100
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
import pandas as pd
import numpy
import tensorflow as tf
from Keras import glove_embedding as embedding
from Keras import util


tf.flags.DEFINE_float("zoneout", 0.2, "Apply zoneout (dropout) to F gate")
tf.flags.DEFINE_integer("max_sequence_length", 1000,
                       "Maximum length of question length")

# Word embeddings
tf.flags.DEFINE_string("word_embedding_path", '',
                       "Where the word embedding vectors are located.")
tf.flags.DEFINE_integer("embedding_vector_dimension", None,
                        "Word embedding vector's dimension.")

# Training
tf.flags.DEFINE_bool("remove_stopwords", True, "Remove stop words")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size")
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.flags.DEFINE_string("optimizer", "nadam",
                       "Optimization method. One of 'adadelta', 'adam', nadam"
                       "'adamax', 'sgd', 'adagrad', 'rmsprop'")
tf.flags.DEFINE_string("raw_train_data", None,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_test_data", None,
                       "Where the raw train data is stored.")

# LSTM model
tf.flags.DEFINE_integer("lstm_out_dimension", 50,
                        "Hidden state dimension (LSTM output vector dimension)")

# Model selection.
tf.flags.DEFINE_string(
    "model", "base_model",
    "Name of a model to run. One of 'base_model', 'bidirectional_rnn', 'qrnn'.")

FLAGS = tf.flags.FLAGS

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

def build_model(lstm_layer_lhs, lstm_layer_rhs, input_sequence_1, input_sequence_2):
    # Square difference
    addition = add([lstm_layer_lhs, lstm_layer_rhs])
    minus_lstm_layer_rhs = Lambda(lambda x: -x)(lstm_layer_rhs)
    merged = add([lstm_layer_lhs, minus_lstm_layer_rhs])
    merged = multiply([merged, merged])

    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    out = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_sequence_1, input_sequence_2], outputs=out)

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

def train(model, train_set, validation_set):
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    csv_logger = CSVLogger('training.log')
    logging = TensorBoard(log_dir='./logs',
                        histogram_freq=0,
                        batch_size=FLAGS.batch_size,
                        write_graph=True,
                        write_grads=False,
                        write_images=False,
                        embeddings_freq=0,
                        embeddings_layer_names=None,
                        embeddings_metadata=None)
    best_model_path = "best_model.h5"
    model_checkpoint = ModelCheckpoint(best_model_path,
                                    save_best_only=True,
                                    save_weights_only=True)
    history = model.fit([train_set[0],train_set[1]],
                    train_set[2],
                    validation_data=([validation_set[0], validation_set[1]], validation_set[2]),
                    epochs=FLAGS.num_epochs,
                    batch_size=FLAGS.batch_size,
                    shuffle=True,
                    callbacks=[early_stopping, model_checkpoint, logging, csv_logger],
                    verbose=1)

    # evaluate model
    train_score = model.evaluate([train_set[0],train_set[1]], train_set[2], verbose=True)
    print("Training:  ", train_score)
    print("--------------------")
    print("First 5 samples validation:", history.history["val_acc"][0:5])
    print("First 5 samples training:", history.history["acc"][0:5])
    print("--------------------")
    print("Last 5 samples validation:", history.history["val_acc"][-5:])
    print("Last 5 samples training:", history.history["acc"][-5:])

def main():
    embedding_layer, labels, question1_list, question2_list, tokenizer = embedding\
        .process_data(FLAGS.word_embedding_path,
                      FLAGS.raw_train_data,
                      FLAGS.embedding_vector_dimension,
                      FLAGS.max_sequence_length)

    lstm_layer = build_lstm_layer()

    # Sequence padding to max sequence length
    input_sequence_1 = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
    embedded_sequences_1 = embedding_layer(input_sequence_1)

    input_sequence_2 = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
    embedded_sequences_2 = embedding_layer(input_sequence_2)

    # Feeding embedded sequence to LSTM layers
    lstm_layer_lhs = lstm_layer(embedded_sequences_1)
    lstm_layer_rhs = lstm_layer(embedded_sequences_2)

    train_data_1, train_data_2 = generate_padded_sequence(question1_list, question2_list, tokenizer)

    # Read test data and do same for test data.
    test = pd.read_csv(FLAGS.raw_test_data)
    test["question1"] = test["question1"].fillna("").apply(util.clean_text) \
        .apply(util.remove_stop_words_and_punctuation)
    test["question2"] = test["question2"].fillna("").apply(util.clean_text) \
        .apply(util.remove_stop_words_and_punctuation)

    test_data_1, test_data_2 = generate_padded_sequence(test["question1"],
                                                        test["question2"],
                                                        tokenizer)

    # Split the data into a training set and a validation set.
    VALIDATION_SPLIT = 0.2
    num_validation_samples = int(VALIDATION_SPLIT * test_data_1.shape[0])

    train_set = [
        train_data_1[:-num_validation_samples],
        train_data_2[:-num_validation_samples],
        labels[:-num_validation_samples]
    ]
    validation_set = [
        train_data_1[-num_validation_samples:],
        train_data_2[-num_validation_samples:],
        labels[-num_validation_samples:]
    ]

    # Building and Training a model
    model = build_model(lstm_layer_lhs, lstm_layer_rhs, input_sequence_1, input_sequence_2)
    train(model, train_set, validation_set)


    # Testing and generating submission csv
    print("Read test.csv file...")
    preds = model.predict([test_data_1, test_data_2], batch_size=FLAGS.batch_size,
                          verbose=1)

    submission = pd.DataFrame({"test_id": test["test_id"], "is_duplicate": preds.ravel()})
    submission.to_csv("preds" + ".csv", index=False)

if __name__ == "__main__":
    main()