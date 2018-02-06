"""I/O utility functions
"""

import os
import json
import csv
import time

import tensorflow as tf

TRAIN_ROOT_FOLDER = './tmp'

def create_train_folder():
    """Create train folder where to put temporary data
    """
    tf.gfile.MakeDirs(TRAIN_ROOT_FOLDER)
    new_dir = '%s/%s' % (TRAIN_ROOT_FOLDER,
                         time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())))
    os.mkdir(new_dir)
    return new_dir

def save_csv(obj, filename):
    """Save to disk
    """
    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        if type(obj) is dict:
            for key, value in obj.items():
                writer.writerow([key, value])
        elif type(obj) is list:
            for vector in obj:
                writer.writerow(vector)

def load_csv(filename):
    """Load from disk
    """
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        return dict(reader)

def save_json(obj, filename):
    """Save to json
    """
    with open(filename, 'w') as json_file:
        json.dump(obj, json_file)

def load_json(filename):
    """Load from disk
    """
    with open(filename, 'r') as json_file:
        return json.load(json_file)
