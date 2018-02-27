import numpy as np
import pandas as pd
import graph
import tensorflow as tf

FREQ_UPPER_BOUND = 100


def create_question_hash(train_df, test_df):
    train_qs = np.dstack([train_df["question1"], train_df["question2"]]).flatten()
    test_qs = np.dstack([test_df["question1"], test_df["question2"]]).flatten()
    all_qs = np.append(train_qs, test_qs)
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


def main():
    tf.flags.DEFINE_string("raw_train_data", None, "Where the raw train data is stored.")
    tf.flags.DEFINE_string("raw_test_data", None, "Where the raw test data is stored.")

    tf.flags.DEFINE_string("path_train_non_nlp_data", None, "Where the raw train data is stored.")
    tf.flags.DEFINE_string("path_test_non_nlp_data", None, "Where the raw test data is stored.")

    flags = tf.flags.FLAGS

    train_df = pd.read_csv(flags.raw_train_data)
    test_df = pd.read_csv(flags.raw_test_data)

    print("Hashing the questions...")
    question_dict = create_question_hash(train_df, test_df)
    train_df = get_hash(train_df, question_dict)
    test_df = get_hash(test_df, question_dict)
    print("Number of unique questions:", len(question_dict))

    print("Calculating K-Core features...")
    all_df = pd.concat([train_df, test_df])
    kcore_dict = graph.get_kcore_dict(all_df)
    train_df = graph.get_kcore_features(train_df, kcore_dict)
    test_df = graph.get_kcore_features(test_df, kcore_dict)
    train_df = convert_to_minmax(train_df, "kcore")
    test_df = convert_to_minmax(test_df, "kcore")

    print("Calculating common neighbor features...")
    neighbors = graph.get_neighbors(train_df, test_df)
    train_df = graph.get_neighbor_features(train_df, neighbors)
    test_df = graph.get_neighbor_features(test_df, neighbors)

    print("Calculating frequency features...")
    frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])),
                                        return_counts=True)))
    train_df = get_freq_features(train_df, frequency_map)
    test_df = get_freq_features(test_df, frequency_map)
    train_df = convert_to_minmax(train_df, "freq")
    test_df = convert_to_minmax(test_df, "freq")

    cols = ["min_kcore",
            "max_kcore",
            "common_neighbor_count",
            "common_neighbor_ratio",
            "min_freq",
            "max_freq"]
    train_df.loc[:, cols].to_csv(flags.path_train_non_nlp_data, index=False)
    test_df.loc[:, cols].to_csv(flags.path_test_non_nlp_data, index=False)


if __name__ == "__main__":
    main()
