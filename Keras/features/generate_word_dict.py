import networkx
import numpy as np
import pandas as pd
import graph
import tensorflow as tf
import os
import nlp
from keras.preprocessing.text import Tokenizer

tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_test_data", None, "Where the raw test data is stored.")
tf.flags.DEFINE_string("save_word_edge_features_path", None,
                       "Where the word_edge_features of train data is going to be saved.")

FLAGS = tf.flags.FLAGS


def remove_rare_words(text, top_words):
    list_word = []
    text = text.split()
    for word in text:
        if word in top_words:
            list_word.append(word)
    return ' '.join(list_word)


def create_word_dict():
    print("Reading train.csv...")
    train = pd.read_csv(FLAGS.raw_train_data)
    print("Cleaning quesitons...")
    train["question1"] = train["question1"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)
    train["question2"] = train["question2"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)

    print("Building question hash...")
    question1_list = np.array(train["question1"])
    question2_list = np.array(train["question2"])

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(np.append(question1_list, question2_list))
    word_freq = tokenizer.word_counts
    word_freq = list(sorted(word_freq.iteritems(),
                            key=lambda x: x[1],
                            reverse=True))
    top_2000_words = word_freq[:2000]

    set_of_words = set(word for word, value in top_2000_words)

    train["question1"] = train["question1"].apply(lambda x: remove_rare_words(x, set_of_words))
    train["question2"] = train["question2"].apply(lambda x: remove_rare_words(x, set_of_words))

    print("Generating ID for words")
    top_words = pd.DataFrame(top_2000_words)[0].drop_duplicates()
    top_words.reset_index(inplace=True, drop=True)

    print("Number of unique words:", len(top_words))
    return top_words, train, set_of_words


def normalize_feature(feature_weight, max_weight):
    return feature_weight/max_weight


def build_graph():
    max_weight = -109000000;
    top_words, dataset, set_of_words = create_word_dict()
    word_dict = pd.Series(top_words.index.values, index=top_words.values).to_dict()
    print (word_dict["best"])
    graph = networkx.MultiGraph()
    graph.add_nodes_from(top_words.index.values)

    for question1, question2, is_duplicate in \
            np.stack((dataset["question1"],
                      dataset["question2"],
                      dataset["is_duplicate"]), axis=-1):
        for word_q1 in question1.split():
            for word_q2 in question2.split():
                edge_weight = (1 if is_duplicate else -1) / 2000000.0
                node_a = word_dict[word_q1]
                node_b = word_dict[word_q2]

                # Make always node_a <= node_b
                if node_a > node_b:
                    temp = node_a
                    node_a = node_b
                    node_b = temp

                if graph.has_edge(node_a, node_b):
                    graph[node_a][node_b][0]['weight'] = graph[node_a][node_b][0]['weight'] \
                                                         + edge_weight
                    max_weight = max(max_weight, graph[node_a][node_b][0]['weight'])

                else:
                    graph.add_edge(node_a, node_b, weight=edge_weight)

    print "maximum weight from edge building"
    return graph, set_of_words, word_dict, dataset


def generate_feature(graph, word_dict, dataset, category):
    # Now compute the actual feature after preprocess.
    feature = []
    max_edge_weight = 1.0
    for question1, question2 in np.stack((dataset["question1"], dataset["question2"]), axis=-1):
        feature_weight = 0.0
        for word_q1 in question1.split():
            for word_q2 in question2.split():
                node_a = word_dict[word_q1]
                node_b = word_dict[word_q2]

                # Make always node_a <= node_b
                if node_a > node_b:
                    temp = node_a
                    node_a = node_b
                    node_b = temp

                if graph.has_edge(node_a, node_b):
                    feature_weight = feature_weight + graph[node_a][node_b][0]['weight']
                    sign = -1 if feature_weight < 0 else 1
                    feature_weight = sign * max(abs(feature_weight), 40000000.0)
        max_edge_weight = max(abs(feature_weight), max_edge_weight)
        feature.append(feature_weight)

    feature = [normalize_feature(feature_weight, max_edge_weight) for feature_weight in feature]
    save_dict = pd.DataFrame({"edge_feature": feature})
    save_path = os.path.join(FLAGS.save_word_edge_features_path+"edge_feature_"+category+".csv")
    save_dict.to_csv(save_path)
    print(max_edge_weight)


def preprocess_test_data(set_of_words):
    print("Reading train.csv...")
    test = pd.read_csv(FLAGS.raw_test_data)
    print("Cleaning quesitons...")
    test["question1"] = test["question1"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)
    test["question2"] = test["question2"].fillna("").apply(lambda x: nlp.clean_text(x, False)) \
        .apply(nlp.remove_stop_words).apply(nlp.word_net_lemmatize)

    test["question1"] = test["question1"].apply(lambda x: remove_rare_words(x, set_of_words))
    test["question2"] = test["question2"].apply(lambda x: remove_rare_words(x, set_of_words))

    return test


def main():
    graph, set_of_words, word_dict, train = build_graph()

    generate_feature(graph, word_dict, train, "train")

    test = preprocess_test_data(set_of_words)

    generate_feature(graph, word_dict, test, "test")


if __name__ == "__main__":
    main()
