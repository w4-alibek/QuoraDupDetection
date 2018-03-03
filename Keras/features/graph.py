from collections import defaultdict
import networkx
import pandas as pd
import numpy as np

NEIGHBOR_UPPER_BOUND = 5
NUMBERS_CORES = 10


def get_kcore_dict(dataset):
    g = networkx.Graph()
    g.add_nodes_from(dataset.qid1)
    edges = list(dataset[["qid1", "qid2"]].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())

    g_nodes = np.array(g.nodes());
    df_output = pd.DataFrame(data=g_nodes, columns=["qid"])
    df_output["kcore"] = 0
    for k in range(2, NUMBERS_CORES + 1):
        ck = networkx.k_core(g, k=k).nodes()
        print("kcore", k)
        df_output.ix[df_output.qid.isin(ck), "kcore"] = k

    return df_output.to_dict()["kcore"]


def get_kcore_features(df, kcore_dict):
    df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
    df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
    return df


def get_neighbors(data_set):
    neighbors = defaultdict(set)
    for q1, q2 in zip(data_set["qid1"], data_set["qid2"]):
        neighbors[q1].add(q2)
        neighbors[q2].add(q1)
    return neighbors


def get_neighbor_features(question_dataset, neighbors):
    common_neighbors = question_dataset.apply(
        lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    min_num_neighbors = question_dataset.apply(
        lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)

    question_dataset["common_neighbor_ratio"] = common_neighbors / min_num_neighbors
    question_dataset["common_neighbor_count"] = common_neighbors.apply(
        lambda x: min(x, NEIGHBOR_UPPER_BOUND))
    return question_dataset