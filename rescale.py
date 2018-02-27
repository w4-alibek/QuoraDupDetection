"""
Rescale test scores to take into account distribution imbalance between training/test set.
https://www.kaggle.com/c/quora-question-pairs/discussion/31179
"""

import csv
import tensorflow as tf

tf.flags.DEFINE_string("file", None, "Submission file")
tf.flags.DEFINE_string("output", None, "Rescaled submission file")

FLAGS = tf.flags.FLAGS


def rescale(x):
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    return a * x / (a * x + b * (1 - x))


with open(FLAGS.output, "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['is_duplicate', 'test_id'])
    with open(FLAGS.file, "r") as original_file:
        next(original_file)
        reader = csv.reader(original_file, delimiter=',')
        for row in reader:
            rescaled = rescale(float(row[0]))
            writer.writerow([rescaled, row[1]])
