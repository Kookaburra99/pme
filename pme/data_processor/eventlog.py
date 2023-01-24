import os
import pandas as pd
from pathlib import Path

from ..utils import DataFrameFields
from stats import get_num_cases, get_num_activities, get_num_resources


class EventlogDataset:
    """
    Custom Dataset for .csv eventlogs (uses Pandas DataFrame)
    """
    filename: str
    directory: Path

    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame

    num_activities: int
    num_resources: int

    def __init__(self, csv_path: str, cv_fold: int = None, read_test: bool = True):
        """
        Creates the EventlogDataset and read the eventlog from the path
        :param csv_path: Path to the .csv file with the full eventlog
        :param cv_fold: Number of fold if a cross-validation fold is read
        :param read_test: Boolean indicating if read test split is necessary
        """

        self.filename = Path(csv_path).stem
        self.directory = Path(csv_path).parent

        self.df_train = self.read_split('train', cv_fold)
        self.df_val = self.read_split('val', cv_fold)
        if read_test:
            self.df_test = self.read_split('test', cv_fold)

        self.num_activities, self.num_resources = self.count_activities_resources(use_test=read_test)

    def read_split(self, split: str, cv_fold: int) -> pd.DataFrame:
        """
        Read the .csv file corresponding to the split
        :param split: Partition to read (train, val or test)
        :param cv_fold: Number of fold if a cross-validation fold is read
        :return: Pandas DataFrame with the eventlog split read
        """

        if cv_fold is not None:
            path_to_eventlog = os.path.join(self.directory, "crossvalidation",
                                            "fold" + str(cv_fold) + "_" + split + "_" + self.filename + ".csv")
        else:
            path_to_eventlog = os.path.join(self.directory, "holdout",
                                            split + "_" + self.filename + ".csv")

        df = pd.read_csv(path_to_eventlog, index_col=0)
        df[DataFrameFields.ACTIVITY_COLUMN] = df[DataFrameFields.ACTIVITY_COLUMN]
        df[DataFrameFields.ACTIVITY_COLUMN] = df[DataFrameFields.ACTIVITY_COLUMN].astype('category')
        df[DataFrameFields.TIMESTAMP_COLUMN] = df[DataFrameFields.TIMESTAMP_COLUMN].astype('datetime64[ns]')
        return df

    def count_activities_resources(self, use_test: bool = False):
        """
        Gets the number of unique activities and resources, if resource column
        exists in the DataFrame. By default, use only train and validation sets
        :param use_test: Boolean indicating if use test set
        :return: Number of unique activities in the eventlog
        """
        all_events = pd.concat([self.df_train, self.df_val])
        if use_test:
            all_events = pd.concat([all_events, self.df_test])

        num_activities = get_num_activities(all_events)
        if DataFrameFields.RESOURCE_COLUMN in all_events.columns:
            num_resources = get_num_resources(all_events)
        else:
            num_resources = None
        return num_activities, num_resources

    def get_num_events(self, split: str) -> int:
        """
        Gets the number of events in the eventlog
        :param split: Partition to read (train, val or test)
        :return: The number of events
        """
        if split == 'train':
            return len(self.df_train.index)
        if split == 'val':
            return len(self.df_val.index)
        if split == 'test':
            return len(self.df_test.index)

    def get_num_cases(self, split: str) -> int:
        """
        Gets the number of cases in the eventlog
        :param split: Partition to read (train, val or test)
        :return: The number of cases
        """
        if split == 'train':
            return get_num_cases(self.df_train)
        if split == 'val':
            return get_num_cases(self.df_val)
        if split == 'test':
            return get_num_cases(self.df_test)

    def add_eoc(self, split: str):
        """
        Add a new especial activity at the end of each case indicating the end of the case.
        The end-of-case identifier is equal to the number of real activities in the process.
        :param split: Partition to read (train, val or test)
        """

        def __add_identifier_at_end_groupby(data: pd.DataFrame):
            data_augment = pd.DataFrame()

            # Group by case
            cases = data.groupby(DataFrameFields.CASE_COLUMN)
            for _, case in cases:
                case = case.reset_index(drop=True)

                new_row = case.iloc[0]
                new_row[DataFrameFields.ACTIVITY_COLUMN] = self.num_activities
                if DataFrameFields.RESOURCE_COLUMN in data.columns:
                    new_row[DataFrameFields.RESOURCE_COLUMN] = self.num_resources
                case = pd.concat([case, new_row])
                case = case.reset_index(drop=True)

                data_augment = pd.concat([data_augment, case])
            return data_augment

        if split == 'train':
            self.df_train = __add_identifier_at_end_groupby(self.df_train)
        if split == 'val':
            self.df_val = __add_identifier_at_end_groupby(self.df_val)
        if split == 'test':
            self.df_test = __add_identifier_at_end_groupby(self.df_test)
