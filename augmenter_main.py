#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Quora data augmenter"""

import operator as op

import numpy as np
import tensorflow as tf

from datasets import quora_dataset
import file_manager

tf.flags.DEFINE_string("data_path", 'data/train.csv',
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("debug", False, "Debug mode")

FLAGS = tf.flags.FLAGS


class _Question(object):

    def __init__(self, qid, question):
        self.qid = int(qid)
        self.question = question

    def __repr__(self):
        return "%d: %s" % (self.qid, self.question)


def _get_sorted_qids(data):
    unique_ids, counts = np.unique(data[:, 0:1], return_counts=True)
    count_dic = dict(zip(unique_ids, counts))
    ids = []
    for qid, _ in sorted(count_dic.items(), key=op.itemgetter(1),
                         reverse=True):
        ids.append(qid)

    return np.array(ids)


def _make_graph_by_qid(qid, data, idset):
    idset.add(qid)
    is_contained_id = np.logical_or(data[:, 0] == qid,
                                    data[:, 1] == qid)
    rest_data = data[np.logical_not(is_contained_id)]

    for row in data[is_contained_id]:
        if row[0] == qid:
            other = row[1]
        else:
            other = row[0]
        rest_data = _make_graph_by_qid(other, rest_data, idset)
    return rest_data


def _make_graph(duplicated_data, sorted_ids):
    graph = {}
    graph_index = 0
    for qid in sorted_ids:
        skip = False
        for idset in graph.values():
            if qid in idset:
                skip = True
        if skip:
            continue

        idset = set()
        duplicated_data = _make_graph_by_qid(qid, duplicated_data, idset)
        graph[graph_index] = idset
        graph_index += 1
    return graph


def _get_data(data_path):
    question_map = {}
    data = []

    for row in quora_dataset(data_path):
        quest1 = _Question(row[1], row[3])
        quest2 = _Question(row[2], row[4])
        if quest1.qid not in question_map:
            question_map[quest1.qid] = quest1
        if quest2.qid not in question_map:
            question_map[quest2.qid] = quest2

        data.append([quest1.qid, quest2.qid, int(row[5])])
        if FLAGS.debug and len(data) > 1000:
            break
    data = np.array(data)
    return data, question_map


def _save_files(data, question_map, graph, output):
    result = [['qid', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']]
    index = 0
    duplicated_idsets = []
    for graph_index in graph:
        sorted_graph = sorted(graph[graph_index])
        for i in range(0, len(sorted_graph)):
            quest1 = question_map[sorted_graph[i]]
            for j in range(i + 1, len(sorted_graph)):
                quest2 = question_map[sorted_graph[j]]
                duplicated_idsets.append([sorted_graph[i], sorted_graph[j]])
                result.append([index, sorted_graph[i], sorted_graph[j],
                               quest1.question, quest2.question, 1])
                index += 1

    for row in data[data[:, 2] == 0]:
        if [row[0], row[1]] in duplicated_idsets:
            continue
        quest1 = question_map[row[0]]
        quest2 = question_map[row[1]]
        result.append([index, row[0], row[1], quest1.question,
                       quest2.question, row[2]])
        index += 1

    file_manager.save_csv(result, output)

def _main():
    print 'Read data.'
    data, question_map = _get_data(FLAGS.data_path)

    print 'Make graph in duplicated questions.'
    duplicated_data = data[data[:, 2] == 1]
    sorted_ids = _get_sorted_qids(duplicated_data)
    graph = _make_graph(duplicated_data, sorted_ids)

    print 'Save augmented data.'
    _save_files(data, question_map, graph, "augmented_dup_data.csv")

if __name__ == '__main__':
    _main()
