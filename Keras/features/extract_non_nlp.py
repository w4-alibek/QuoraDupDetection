import numpy as np
import pandas as pd
import graph
import tensorflow as tf

FREQ_UPPER_BOUND = 100
FEATURE_COLS = ["min_kcore",
                "max_kcore",
                "common_neighbor_count",
                "common_neighbor_ratio",
                "min_freq",
                "max_freq"]


def create_question_hash(data_set):
    all_qs = np.dstack([data_set["question1"], data_set["question2"]]).flatten()
    all_qs = pd.DataFrame(all_qs)[0].drop_duplicates()
    all_qs.reset_index(inplace=True, drop=True)
    question_dict = pd.Series(all_qs.index.values, index=all_qs.values).to_dict()
    return question_dict


def get_hash(df, hash_dict):
    df["qid1"] = df["question1"].map(hash_dict)
    df["qid2"] = df["question2"].map(hash_dict)
    return df.drop(["question1", "question2"], axis=1)


def convert_to_minmax(df, col):
    sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T)
    df["min_" + col] = sorted_features[:, 0]
    df["max_" + col] = sorted_features[:, 1]
    return df.drop([col + "1", col + "2"], axis=1)


def get_freq_features(df, frequency_map):
    df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    return df

##########################
###GENERATING FEATURES.###
##########################


def generate_features_for_train(flags):
    print("Reading train.csv...")
    train_data_features = pd.read_csv(flags.raw_train_data)
    print("Hashing the questions...")
    train_question_dict = create_question_hash(train_data_features)
    train_data_features = get_hash(train_data_features, train_question_dict)
    print("Number of unique train questions:", len(train_question_dict))

    print("Calculating K-Core features for train...")
    kcore_dict_train = graph.get_kcore_dict(train_data_features)
    train_data_features = graph.get_kcore_features(train_data_features, kcore_dict_train)
    train_data_features = convert_to_minmax(train_data_features, "kcore")

    print("Calculating common neighbor features for train...")
    neighbors_train = graph.get_neighbors(train_data_features)
    train_data_features = graph.get_neighbor_features(train_data_features, neighbors_train)

    print("Calculating frequency features for train...")
    frequency_map_train = dict(zip(*np.unique(np.vstack((train_data_features["qid1"],
                                                         train_data_features["qid2"])),
                                              return_counts=True)))
    train_data_features = get_freq_features(train_data_features, frequency_map_train)
    train_data_features = convert_to_minmax(train_data_features, "freq")

    print("Saving train data set features...")
    train_data_features.loc[:, FEATURE_COLS].to_csv(flags.path_train_non_nlp_data, index=False)


def generate_features_for_test(flags):
    print("Reading test.csv...")
    test_data_features = pd.read_csv(flags.raw_test_data)
    print("Hashing the questions...")
    test_question_dict = create_question_hash(test_data_features)
    test_data_features = get_hash(test_data_features, test_question_dict)
    print("Number of unique test questions:", len(test_question_dict))

    print("Calculating K-Core features for test...")
    kcore_dict_test = graph.get_kcore_dict(test_data_features)
    test_data_features = graph.get_kcore_features(test_data_features, kcore_dict_test)
    test_data_features = convert_to_minmax(test_data_features, "kcore")

    print("Calculating common neighbor features for test...")
    neighbors_test = graph.get_neighbors(test_data_features)
    test_data_features = graph.get_neighbor_features(test_data_features, neighbors_test)

    print("Calculating frequency features for test...")
    frequency_map_test = dict(zip(*np.unique(np.vstack((test_data_features["qid1"],
                                                        test_data_features["qid2"])),
                                             return_counts=True)))

    test_data_features = get_freq_features(test_data_features, frequency_map_test)
    test_data_features = convert_to_minmax(test_data_features, "freq")

    print("Saving test data set features...")
    test_data_features.loc[:, FEATURE_COLS].to_csv(flags.path_test_non_nlp_data, index=False)


def main():
    tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data is stored.")
    tf.flags.DEFINE_string("raw_test_data", None, "Where the raw test data is stored.")

    tf.flags.DEFINE_string("path_train_non_nlp_data", None, "Where the raw train data is stored.")
    tf.flags.DEFINE_string("path_test_non_nlp_data", None, "Where the raw test data is stored.")

    flags = tf.flags.FLAGS

    generate_features_for_train(flags);
    print ("#########################################################")
    print ("#######################FINISH############################")
    print ("#########################################################")
    generate_features_for_test(flags);
    print ("#########################################################")
    print ("#######################FINISH############################")
    print ("#########################################################")


if __name__ == "__main__":
    main()
