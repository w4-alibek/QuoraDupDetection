"""Convenient functions for loading datasets
"""

import tensorflow as tf
import util


def quora_dataset(clean_data, *filenames):
    """Row generator for quora kaggle dataset

    Some lines are broken in the dataset, this function automatically takes care of it.
    Removes double quotes in questions, eg. ""foo""

    """
    for filename in filenames:
        with tf.gfile.GFile(filename, "r") as gfile:
            lines = gfile.read().split("\n")
            broken_line = None

            # skip first line (column labels)
            quora_data = iter(lines)
            next(quora_data)

            for line in quora_data:
                line = line.strip().replace('""', '')

                if broken_line is not None:
                    line = broken_line + line

                row = [x for x in line.split('\"') if (x != ',' and x != '')]

                if len(row) != 6:
                    broken_line = line
                    continue

                broken_line = None

                # remove unwanted characters
                if clean_data:
                    row[3] = util.remove_stop_words_and_punctuation(row[3])
                    row[4] = util.remove_stop_words_and_punctuation(row[4])

                if len(row[3]) > 0 and len(row[4]) > 0:
                    yield row

def glove_dataset(dimension, *filenames):
    """Generator for Glove embeddings
    """
    for filename in filenames:
        with tf.gfile.GFile(filename, "r") as gfile:
            lines = gfile.read().split("\n")

            for line in lines:
                line = line.split(" ", 1 + dimension)
                yield line
