import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from ..utils import DataFrameFields


def make_holdout(csv_path: str, train_size: float = 80,
                 val_size_from_train: float = 20, splits_path: str = None):
    """
    Create the train-val-test splits and store them
    :param csv_path: Full path to the CSV file with the dataset
    :param train_size: Percentage of the data for the training partition
    (the test partition is the remaining percentage). Number between 1 and 100
    :param val_size_from_train: Percentage of the training partition reserved
    for validation. Number between 1 and 100
    :param splits_path: Full path where CSV splits will be written
    """
    full_df = pd.read_csv(csv_path)

    df_groupby = full_df.groupby(DataFrameFields.CASE_COLUMN, sort=False)
    cases = [case for _, case in df_groupby]

    real_val_size = (train_size / 100) * (val_size_from_train / 100)
    real_train_size = (train_size / 100) - real_val_size

    first_cut = round(len(cases) * real_train_size)
    second_cut = round(len(cases) * (real_train_size + real_val_size))

    train_cases = cases[:first_cut]
    val_cases = cases[first_cut:second_cut]
    test_cases = cases[second_cut:]

    train_df = pd.concat(train_cases)
    val_df = pd.concat(val_cases)
    test_df = pd.concat(test_cases)

    filename = Path(csv_path).stem + ".csv"
    write_path = splits_path if splits_path else str(Path(full_csv_path).parent) + '/holdout/'

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    train_df.to_csv(write_path + 'train_' + filename)
    val_df.to_csv(write_path + 'val_' + filename)
    test_df.to_csv(write_path + 'test_' + filename)


def make_crossvalidation(csv_path: str, num_folds: int = 5, val_size_from_train: float = 20,
                         splits_path: str = None, seed: int = 21):
    """
    Create the k-fold cross-validation and store the folds
    :param csv_path: Full path to the CSV file with the dataset
    :param num_folds: Number of folds in the cross-validation
    :param val_size_from_train: Percentage of the training partition reserved
    for validation. Number between 1 and 100
    :param splits_path: Full path where CSV splits will be written
    :param seed: Seed to set the random state and get reproducibility
    """
    full_df = pd.read_csv(csv_path)

    unique_case_ids = list(full_df[DataFrameFields.CASE_COLUMN].unique())
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    indexes = sorted(unique_case_ids)
    splits = kfold.split(indexes)

    filename = Path(csv_path).stem + ".csv"
    write_path = splits_path if splits_path else str(Path(csv_path).parent) + '/crossvalidation'

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    fold = 0
    for train_index, test_index in splits:
        val_cut = round(len(train_index) * ((100 - val_size_from_train) / 100))

        val_index = train_index[val_cut:]
        train_index = train_index[:val_cut]

        train_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == train_g] for train_g in train_index]
        val_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == val_g] for val_g in val_index]
        test_groups = [full_df[full_df[DataFrameFields.CASE_COLUMN] == test_g] for test_g in test_index]

        train_df = pd.concat(train_groups)
        val_df = pd.concat(val_groups)
        test_df = pd.concat(test_groups)

        train_df.to_csv(write_path + "/fold" + str(fold) + "_train_" + filename)
        val_df.to_csv(write_path + "/fold" + str(fold) + "_val_" + filename)
        test_df.to_csv(write_path + "/fold" + str(fold) + "_test_" + filename)

        fold += 1
