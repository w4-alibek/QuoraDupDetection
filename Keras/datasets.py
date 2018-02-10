"""Convenient functions for loading datasets
"""

import tensorflow as tf

from dupDetection import util

question_index_trainset_1 = 3
question_index_trainset_2 = 4
question_index_testset_1 = 1
question_index_testset_2 = 2
label_index_trainset = 5
label_index_testset = -1

def quora_dataset(clean_data, filename, set_type):
    """Row generator for quora kaggle dataset

    Some lines are broken in the dataset, this function automatically takes care of it.
    Removes double quotes in questions, eg. ""foo""

    """
    q1_idx = q2_idx = 0
    if set_type == 'test':
        q1_idx = question_index_testset_1
        q2_idx = question_index_testset_2
        label_idx = label_index_testset
    elif set_type == 'train':
        q1_idx = question_index_trainset_1
        q2_idx = question_index_trainset_2
        label_idx = label_index_trainset
    else:
        raise Exception('Set type must be train or test')

    with tf.gfile.GFile(filename, "r") as gfile:
        lines = gfile.read().decode("utf-8")
        lines = lines.split('\r\n');
        broken_line = None
        tmp = 'sdfasdf,asdfasdfasdf,asdfasdf'
        tmp = tmp.split(',');
        # skip first line (column labels)
        quora_data = iter(lines)
        next(quora_data)

        for line in quora_data:
            line = line.strip().replace('""', '')

            if broken_line is not None:
                line = broken_line + line

            row = [x for x in line.split('\"') if (x != ',' and x != '')]

            if len(row) == 0:
                continue

            if set_type == 'train' and len(row) != 6:
                broken_line = line
                continue

            broken_line = None

            # remove unwanted characters
            if clean_data:
                row[q1_idx] = util.remove_stop_words_and_punctuation(row[q1_idx])
                row[q2_idx] = util.remove_stop_words_and_punctuation(row[q2_idx])

            if len(row[q1_idx]) > 0 and len(row[q2_idx]) > 0:
                yield row, q1_idx, q2_idx, label_idx


def glove_dataset(dimension, *filenames):
    """Generator for Glove embeddings
    """
    for filename in filenames:
        with tf.gfile.GFile(filename, "r") as gfile:
            lines = gfile.read().decode("utf-8").split("\n")

            for line in lines:
                line = line.split(" ", 1 + dimension)
                yield line
