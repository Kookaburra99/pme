import os
from pathlib import Path
import pandas as pd
import pm4py

from ..utils import DataFrameFields, XESFields


def get_datasets_list(path: str, batch_mode: bool) -> list:
    """
    Get list of paths to datasets to be processed
    :param path: Path to the dataset or folder
    :param batch_mode: If batch mode is used or only one dataset
    :param logger: DataProcessorLogger to print the stats
    :return: A list of the path to datasets
    """

    dataset_list = []
    if batch_mode:
        files = os.listdir(path)
        for dataset in files:
            dataset_list.append(os.path.join(path, dataset))
    else:
        dataset_list.append(path)

    return dataset_list


def __select_output_columns(activity: bool, timestamp: bool,
                            resource: bool) -> dict:
    """
    Select and keep from the dataset only the specified columns
    :param activity: Boolean indicating if keep the activity column
    :param timestamp: Boolean indicating if keep the timestamp column
    :param resource: Boolean indicating if keep the resource column
    :param others: List of strings with the names of others XES log columns
    :return: Dictionary with the specified columns
    """

    output_columns = {XESFields.CASE_COLUMN: DataFrameFields.CASE_COLUMN}
    if activity:
        output_columns[XESFields.ACTIVITY_COLUMN] = DataFrameFields.ACTIVITY_COLUMN
    if timestamp:
        output_columns[XESFields.TIMESTAMP_COLUMN] = DataFrameFields.TIMESTAMP_COLUMN
    if resource:
        output_columns[XESFields.RESOURCE_COLUMN] = DataFrameFields.RESOURCE_COLUMN

    return output_columns


def __get_idx_attribute(attr: pd.Series) -> list:
    """
    Convert dataset column (pd.Series) type to categorical
    :param attr: Pandas Series of the column in the dataset
    :return: List with the values in the new categorical type
    """
    unique_attr = attr.unique()
    attr_dict = {act: idx for idx, act in enumerate(unique_attr)}
    attr_idx = list(map(lambda x: attr_dict[x], attr.values))

    return attr_idx


def convert_xes_to_csv(xes_path: str, use_act: bool = True, use_time: bool = True,
                       use_res: bool = True, csv_path: str = None) -> str:
    """
    Convert the XES file with the dataset to a CSV format file
    :param xes_path: Full path to the XES file
    :param use_act: Boolean indicating if use activity column
    :param use_time: Boolean indicating if use timestamp column
    :param use_res: Boolean indicating if use resource column
    :param csv_path: Path where the CSV file will be stored
    :return: Full path to the CSV file
    """

    log = pm4py.read_xes(xes_path)
    df_log = pm4py.convert_to_dataframe(log)

    # Get real activities
    if XESFields.LIFECYCLE_COLUMN in df_log:
        unique_lifecycle = df_log[XESFields.LIFECYCLE_COLUMN].unique()
        if len(unique_lifecycle) > 1:
            df_log.loc[:, XESFields.ACTIVITY_COLUMN] = df_log[XESFields.ACTIVITY_COLUMN].astype(str) + "+" \
                                                + df_log[XESFields.LIFECYCLE_COLUMN]

    # Correct timestamp format
    if XESFields.TIMESTAMP_COLUMN in df_log:
        df_log.loc[:, XESFields.TIMESTAMP_COLUMN] = pd.to_datetime(
            df_log[XESFields.TIMESTAMP_COLUMN], utc=True)

    output_columns = __select_output_columns(use_act, use_time, use_res)

    # Select relevant columns
    if output_columns:
        df_log = df_log[list(output_columns.keys())]

        if XESFields.CASE_COLUMN in output_columns:
            df_log.loc[:, XESFields.CASE_COLUMN] = __get_idx_attribute(
                df_log[XESFields.CASE_COLUMN])
            df_log[XESFields.CASE_COLUMN].astype(str)

        if XESFields.ACTIVITY_COLUMN in output_columns:
            df_log.loc[:, XESFields.ACTIVITY_COLUMN] = __get_idx_attribute(
                df_log[XESFields.ACTIVITY_COLUMN])
            df_log[XESFields.ACTIVITY_COLUMN].astype(str)

        if XESFields.RESOURCE_COLUMN in output_columns:
            df_log.loc[:, XESFields.RESOURCE_COLUMN] = __get_idx_attribute(
                df_log[XESFields.RESOURCE_COLUMN])
            df_log[XESFields.RESOURCE_COLUMN].astype(str)

        df_log.rename(columns=output_columns, inplace="True")

    # Write in CSV
    csv_file = Path(xes_path).stem.split(".")[0] + ".csv"
    if csv_path:
        csv_path = Path(csv_path)
    else:
        csv_path = Path(os.getcwd())
    full_csv_path = os.path.join(csv_path, csv_file)

    df_log.to_csv(full_csv_path, index=False)

    return full_csv_path
