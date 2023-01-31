import pandas as pd
import networkx as nx
import torch
import math
import pm4py


class XESFields:
    """
    Supported xes fields that may be present in a xes log
    """
    CASE_COLUMN = "case:concept:name"
    ACTIVITY_COLUMN = "concept:name"
    TIMESTAMP_COLUMN = "time:timestamp"
    LIFECYCLE_COLUMN = "lifecycle:transition"
    RESOURCE_COLUMN = "org:resource"


class DataFrameFields:
    """
    Formatted column names
    """
    CASE_COLUMN = "CaseID"
    ACTIVITY_COLUMN = "Activity"
    TIMESTAMP_COLUMN = "Timestamp"
    RESOURCE_COLUMN = "Resource"


def get_emb_size_power_of_two(num_categories: int) -> int:
    """
    Calculate the higher power of two, lower than num_categories
    :param num_categories: Number of activities
    """
    exp = int(math.log(num_categories, 2))

    return 2 ** exp


def get_device(gpu: bool = True) -> torch.device:
    """
    Gets the device available (CPU or GPU)
    :param gpu: If True, try to run on the GPU, else try on the CPU
    """
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def get_cases_as_sentences(data: pd.DataFrame) -> list[list]:
    """
    Tokenize activities and preprocess cases as sentences in NLP
    :param data: Pandas DataFrame with the cases
    :return: A list of lists, each of which contains the activities of each case
    """

    list_cases = []

    df_cases = data.groupby(DataFrameFields.CASE_COLUMN)
    for _, df_case in df_cases:
        list_cases.append(df_case[DataFrameFields.ACTIVITY_COLUMN].tolist())

    return list_cases


def get_process_graph(data: pd.DataFrame, num_activities: int) -> nx.Graph:
    """
    Generate the Networkx Graph of the process from the eventlog
    :param data: Pandas DataFrame with the cases
    :param num_activities: Number of unique activities in the process
    :return: Networkx Graph of the process
    """

    dfg, sa, ea = pm4py.discovery.discover_dfg_typed(data,
                                                     activity_key=DataFrameFields.ACTIVITY_COLUMN,
                                                     timestamp_key=DataFrameFields.TIMESTAMP_COLUMN,
                                                     case_id_key=DataFrameFields.CASE_COLUMN)

    G = nx.Graph()
    for pairs, counts in dfg.items():
        G.add_edge(int(pairs[0]), int(pairs[1]))

    for i in range(num_activities):
        if i not in G.nodes:
            G.add_node(i)
    return G


def get_process_weitghted_digraph(data: pd.DataFrame, num_activities: int) -> nx.DiGraph:
    """
    Generate the weighted Networkx DiGraph of the process from the eventlog.
    The weights of the edges depend on the number of times that the both activities
    represented by the nodes appear together in the eventlog.
    :param data: Pandas DataFrame with the cases
    :param num_activities: Number of unique activities in the process
    :return: Networkx DiGraph of the process
    """

    dfg, sa, ea = pm4py.discovery.discover_dfg_typed(data,
                                                     activity_key=DataFrameFields.ACTIVITY_COLUMN,
                                                     timestamp_key=DataFrameFields.TIMESTAMP_COLUMN,
                                                     case_id_key=DataFrameFields.CASE_COLUMN)
    dict_acts_count = {}
    for pairs, counts in dfg.items():
        if int(pairs[0]) in dfg:
            dict_acts_count[int(pairs[0])] += counts
        else:
            dict_acts_count[int(pairs[0])] = counts

    G = nx.DiGraph()
    for pairs, counts in dfg.items():
        G.add_edge(int(pairs[0]), int(pairs[1]), weight=counts/dict_acts_count[int(pairs[0])])

    for i in range(num_activities):
        if i not in G.nodes:
            G.add_node(i)
    return G
