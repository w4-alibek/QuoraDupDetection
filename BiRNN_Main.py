#!/usr/bin/env python

#pylint: disable-msg=too-many-arguments
#pylint: disable-msg=too-many-locals

"""Load data and run training
"""

import os
import datetime
import random
import numpy as np
import tensorflow as tf

import word_to_id_builder
import word_embeddings_builder
import file_manager
import BiRNN_model_builder

# np.set_printoptions(threshold=np.nan)

tf.flags.DEFINE_string("processed_train_data", None,
                       "Where the train data with computed features are stored.")
tf.flags.DEFINE_string("word_to_id_path", None,
                       "Where the word_to_id is stored.")
tf.flags.DEFINE_string("raw_train_data", 'data/train.csv',
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_test_data", 'data/test.csv',
                       "Where the raw train data is stored.")

# Word embeddings
tf.flags.DEFINE_string("glove_by_word_id_path", None,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("glove_path", '',
                       "Where the glove embeddings are stored.")
tf.flags.DEFINE_integer("glove_dimension", None,
                        "Where the raw train data is stored.")

# Training
tf.flags.DEFINE_bool("remove_stopwords", True, "Remove stop words")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size")
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.flags.DEFINE_string("optimizer", "adagrad",
                       "Optimization method. One of 'adadelta', 'adam', "
                       "'sgd', 'adagrad', 'rmsprop'")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")

# LSTM model
tf.flags.DEFINE_integer("lstm_out_dimension", 50,
                        "Hidden state dimension (LSTM output vector dimension)")
tf.flags.DEFINE_integer("num_steps", 5,
                        "Number of time steps for LSTM")

# Parameters for FC layer.
tf.flags.DEFINE_integer("fc_out_dimension", 1024,
                        "Output dimension for fully connected layer.")

#TODO(anthony) balance data
#TODO(anthony) truncated backprop

FLAGS = tf.flags.FLAGS


def eval(sess, model, dataset, embeddings):
    """Evaluate the model on dataset

    :param sess:
    :param model:
    :param dataset:
    :return:
    """
    total_test_accuracy = 0
    count = 0
    for test_question1, test_seqlen1, test_question2, test_seqlen2, test_labels in next(dataset):
        test_accuracy = sess.run(model['accuracy'],
                                 feed_dict={model['input_placeholders'][0]: test_question1,
                                            model['input_placeholders'][1]: test_question2,
                                            model['labels_placeholder']: test_labels,
                                            model['seqlen'][0]: test_seqlen1,
                                            model['seqlen'][1]: test_seqlen2,
                                            model['embeddings']: embeddings['glove'],
                                            model['is_training']: False})
        total_test_accuracy += test_accuracy
        count += 1

    return total_test_accuracy / count


def train(data, graph, model, embeddings, train_folder):
    """Train
    """
    train_sess = tf.Session(graph=graph)
    summary_writer = tf.summary.FileWriter(train_folder)

    with train_sess.graph.as_default():
        train_sess.run(tf.global_variables_initializer(),
                       feed_dict={model['embeddings']: embeddings['glove']})
        saver = tf.train.Saver()
        merged_summary = tf.summary.merge_all() # for tensorboard visualization

    summary_writer.add_graph(train_sess.graph)

    step = 0
    for idx, epoch in enumerate(data['train']):
        print("\nEPOCH", idx)

        total_train_accuracy = 0
        nbatch = 0

        total_loss = 0
        for batch_id, (question1, seqlen1, question2, seqlen2, labels) in enumerate(epoch):

            feed_dict = {model['input_placeholders'][0]: question1,
                         model['input_placeholders'][1]: question2,
                         model['labels_placeholder']: labels,
                         model['seqlen'][0]: seqlen1,
                         model['seqlen'][1]: seqlen2,
                         model['embeddings']: embeddings['glove'],
                         model['is_training']: True}

            if batch_id % 1 == 0:
                # get summary for tensorboard
                _, loss, train_accuracy, summary = \
                    train_sess.run([model['train_step'],
                                    model['loss'],
                                    model['accuracy'],
                                    merged_summary],
                                   feed_dict)
                summary_writer.add_summary(summary, step)
                step += 1
                if batch_id % 200 == 0:
                    print("step %d, training accuracy %g" % (batch_id, train_accuracy))
            else:
                _, loss, train_accuracy = \
                    train_sess.run([model['train_step'],
                                    model['loss'],
                                    model['accuracy']],
                                   feed_dict)

            total_train_accuracy += train_accuracy
            total_loss += loss
            nbatch += 1

        print("Training accuracy %g, total loss %g" % (total_train_accuracy / nbatch, total_loss / nbatch))
        checkpoint_file = os.path.join(train_folder, 'model.ckpt')
        saver.save(train_sess, checkpoint_file, global_step=idx)

        test_accuracy = eval(train_sess, model, data['test'], embeddings)
        print("Testing accuracy %g" % test_accuracy)
    train_sess.close()


def max_length(samples):
    """Finds the maximum length of a question
    """
    return max(map(lambda s: max(len(s['q1']['tokens']), len(s['q2']['tokens'])), samples))


def build_batch_generator(samples, batch_size, question_max_length):
    sample_size = len(samples)
    # TODO(chungshik): Handle the last chunk.
    num_batch = sample_size / batch_size
    random.shuffle(samples)

    for j in range(num_batch):
        data_q1 = np.zeros([batch_size, question_max_length], dtype=np.int32)
        data_q2 = np.zeros([batch_size, question_max_length], dtype=np.int32)
        seqlen1 = np.zeros([batch_size], dtype=np.int32)
        seqlen2 = np.zeros([batch_size], dtype=np.int32)

        chunk_samples = samples[j * batch_size : (j + 1) * batch_size]
        for idx, sample in enumerate(chunk_samples):
            len1 = len(sample['q1']['tokens'])
            len2 = len(sample['q2']['tokens'])
            data_q1[idx][:len1] = sample['q1']['tokens']
            data_q2[idx][:len2] = sample['q2']['tokens']
            seqlen1[idx] = len1
            seqlen2[idx] = len2

        labels = [sample['label'] for sample in chunk_samples]

        yield data_q1, seqlen1, data_q2, seqlen2, labels


def build_train_data_epoch_generator(train_data, num_epochs, batch_size, question_max_length):
    """Build data generator
    """
    for i in range(num_epochs):
        yield build_batch_generator(train_data, batch_size, question_max_length)


def build_test_data_generator(train_data, batch_size, question_max_length):
    while True:
        yield build_batch_generator(train_data, batch_size, question_max_length)

def load_and_process_data(train_folder):
    if not FLAGS.word_to_id_path:
        # find all words
        print "Find all words in corpuses..."
        words = {}
        # TODO(chungshik): Commented the following line temporarily given that
        # there's no reader for test.csv yet.
        ## word_to_id_builder.read_quora_words(words, FLAGS.raw_test_data)
        word_to_id_builder.read_quora_words(words, FLAGS.raw_train_data)
        word_to_id_builder.read_glove_words(words, FLAGS.glove_path)

        # build index
        print "Build word to id dictionary..."
        word_to_id = word_to_id_builder.build_word_to_id(words)

        # for later use
        file_manager.save_csv(word_to_id, train_folder + '/word_to_id.csv')
    else:
        print "Load word to id..."
        word_to_id = file_manager.load_csv(FLAGS.word_to_id_path)
        word_to_id = dict([a, int(x)] for a, x in word_to_id.iteritems())   # cast dict values to int

    # map tokens in question pairs to their word id
    if not FLAGS.processed_train_data:
        print "Build processed train data..."
        processed_train_data = word_to_id_builder.build_tokenized_samples(
            FLAGS.raw_train_data, word_to_id, FLAGS.remove_stopwords)
        file_manager.save_json(processed_train_data, train_folder + '/processed_train_data.json')
    else:
        print "Load processed train data..."
        processed_train_data = file_manager.load_json(FLAGS.processed_train_data)

    # Glove
    if not FLAGS.glove_by_word_id_path:
        print "Build Glove word embeddings..."
        glove_embeddings = word_embeddings_builder.build_glove_embeddings(
            word_to_id, FLAGS.glove_path, FLAGS.glove_dimension)

        np.save(train_folder + '/glove_embeddings_by_word_id.npy', glove_embeddings)
    else:
        print "Load Glove word embeddings..."
        glove_embeddings = np.load(FLAGS.glove_by_word_id_path).astype(np.float32)

    return word_to_id, processed_train_data, glove_embeddings


def main(_):
    """Main
    """
    start_time = datetime.datetime.now()

    train_folder = file_manager.create_train_folder()
    print "Train folder: " + train_folder

    word_to_id, processed_train_data, glove_embeddings = load_and_process_data(train_folder)

    question_max_length = max_length(processed_train_data)

    # Split data to train and validation (named as 'test' here for now) set.
    num_samples = len(processed_train_data)
    split_point = int(num_samples * 0.8)
    train_data = processed_train_data[:split_point]
    test_data = processed_train_data[split_point:num_samples]

    train_data_generator = build_train_data_epoch_generator(
        train_data, FLAGS.num_epochs, FLAGS.batch_size, question_max_length)
    test_data_generator = build_test_data_generator(
        test_data, FLAGS.batch_size, question_max_length)

    train_graph = tf.Graph()

    with train_graph.as_default():
        train_model = BiRNN_model_builder.build(
            {'glove': np.asarray(glove_embeddings)}, FLAGS.lstm_out_dimension,
            FLAGS.fc_out_dimension, question_max_length, FLAGS.batch_size,
            FLAGS.learning_rate, FLAGS.optimizer)

    train({'train': train_data_generator, 'test': test_data_generator},
          train_graph,
          train_model,
          {'glove': glove_embeddings},
          train_folder)

    print "Total time: " + str((datetime.datetime.now() - start_time).total_seconds())


if __name__ == "__main__":
    tf.app.run()