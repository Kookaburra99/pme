import pandas as pd

from ..utils import DataFrameFields


def get_num_cases(data: pd.DataFrame) -> int:
    """
    Get the number of execution cases in the process eventlog
    :param data: Pandas DataFrame with the dataset
    :return: The number of unique cases
    """
    num_cases = data[DataFrameFields.CASE_COLUMN].nunique()
    return num_cases


def get_num_activities(data: pd.DataFrame) -> int:
    """
    Get the number of unique activities in the process eventlog
    :param data: Pandas DataFrame with the dataset
    :return: The number of unique activities
    """
    num_activities = data[DataFrameFields.ACTIVITY_COLUMN].nunique()
    return num_activities


def get_num_resources(data: pd.DataFrame) -> int:
    """
    Get the number of unique resources in the process eventlog
    :param data: Pandas DataFrame with the dataset
    :return: The number of unique resources
    """
    num_resources = data[DataFrameFields.RESOURCE_COLUMN].nunique()
    return num_resources


def get_case_lens(data: pd.DataFrame) -> (int, int, int):
    """
    Get the average, maximum and minimum case length in the process eventlog
    :param data: Pandas DataFrame with the dataset
    :return: The average case length, the max case length and the min case length
    """
    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    avg_case_len = cases[DataFrameFields.ACTIVITY_COLUMN].count().mean()
    max_case_len = cases[DataFrameFields.ACTIVITY_COLUMN].count().max()
    min_case_len = cases[DataFrameFields.ACTIVITY_COLUMN].count().min()
    return avg_case_len, max_case_len, min_case_len


def get_num_variants(data: pd.DataFrame) -> int:
    """
    Get the number of different traces (variants) in the process eventlog
    :param data: Pandas DataFrame with the dataset
    :return: Number of variants (unique sequences of activities)
    """
    data['Activity'] = data['Activity'].astype(str)
    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    num_variants = cases['Activity'].agg("->".join).nunique()
    return num_variants


def get_top_variants(data: pd.DataFrame, top: int = 5) -> dict:
    """
    Get the top most repeated variants
    :param data: Pandas DataFrame with the dataset
    :param top: Number of variants to show in the top
    :return: Dictionary with the top repeated variants and their count
    """
    data['Activity'] = data['Activity'].astype(str)
    cases = data.groupby(DataFrameFields.CASE_COLUMN)
    variants = cases['Activity'].agg("->".join)
    top = len(variants) if top > len(variants) else top
    top_variants = variants[:top].value_counts().to_dict()
    return top_variants
