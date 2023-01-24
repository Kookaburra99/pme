# PME (Process Mining Embeddings)
A package to train and generate activity-level embeddings for process mining

## Authors
- [@Kookaburra99](https://github.com/Kookaburra99)

## Installation
Install the PME package with pip
```bash
$> pip install pme
```

## Requirements
- Pandas
- Scikit-Learn
- PM4PY
- Torch
- Gensim
- KarateClub


## Documentation
The package has three main modules:

### data_processor
It allows to read datasets in .XES format, convert them to .CSV format, save them,
display their main characteristics (number of events, unique activities and 
resources...), perform holdout or cross-validation partitioning and define the 
objects to store the eventlog data and use them with the rest of the package 
functions.

#### eventlog
- ***EventlogDataset***: Class storing the training, validation and test partitions of a 
given dataset and its main features useful for training embeddings.
  - **Attributes**:
    - *filename*: Name of the dataset.
    - *directory*: Path to the folder where the dataset is stored.
    - *df_train*: Pandas DataFrame of the train partition of the dataset.
    - *df_val*: Pandas DataFrame of the validation partition of the dataset.
    - *df_test*: Pandas DataFrame of the test partition of the dataset.
    - *num_activities*: Number of unique activities in the eventlog.
    - *num_resources*: Number of unique resources in the eventlog.
  - **Constructor parameters**:
    - *csv_path*: Path to the .csv file with the full eventlog.
    - *cv_fold*: Number of fold if a cross-validation fold is read. Default: None.
    - *read_test*: Boolean indicating if read test split is necessary. Default: True.

#### split
- ***make_holdout(csv_path: str, train_size: float = 80, 
val_size_from_train: float = 20, splits_path: str = None)***:
Create the train-val-test splits and store them.
  - **Parameters**:
    - *csv_path*: Full path to the CSV file with the dataset.
    - *train_size*: Percentage of the data for the training partition 
    (the test partition is the remaining percentage). Number between 1 and 100.
    - *val_size_from_train*: Percentage of the training partition reserved 
    for validation. Number between 1 and 100.
    - *splits_path*: Full path where CSV splits will be written.
- ***make_crossvalidation(csv_path: str, num_folds: int = 5, val_size_from_train: float = 20, 
splits_path: str = None, seed: int = 21)***:
Create the k-fold cross-validation and store the folds.
  - **Parameters**:
    - *csv_path*: Full path to the CSV file with the dataset.
    - *num_folds*: Number of folds in the cross-validation.
    - *val_size_from_train*: Percentage of the training partition reserved 
    for validation. Number between 1 and 100.
    - *splits_path*: Full path where CSV splits will be written.
    - *seed*: Seed to set the random state and get reproducibility.

#### stats
- ***get_num_cases(data: pd.DataFrame) -> int***:
Get the number of execution cases in the process eventlog.
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
  - **Return**: The number of unique cases.
- ***get_num_activities(data: pd.DataFrame) -> int***:
Get the number of unique activities in the process eventlog.
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
  - **Return**: The number of unique activities.
- ***get_num_resources(data: pd.DataFrame) -> int***:
Get the number of unique resources in the process eventlog.
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
  - **Return**: The number of unique resources.
- ***get_case_lens(data: pd.DataFrame) -> (int, int, int)***:
Get the average, maximum and minimum case length in the process eventlog.
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
  - **Return**: The average case length, the max case length.
and the min case length.
- ***get_num_variants(data: pd.DataFrame) -> int***:
Get the number of different traces (variants) in the process eventlog.
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
  - **Return**: Number of variants (unique sequences of activities).
- ***get_top_variants(data: pd.DataFrame, top: int = 5) -> dict***:
Get the top most repeated variants
  - **Parameters**:
    - *data*: Pandas DataFrame with the dataset.
    - *count*: Number of variants to show in the top.
  - **Return**: Dictionary with the top repeated variants and their count.

#### xes
- ***get_datasets_lists(path: str, batch_mode: bool) -> list***:
Get list of paths to datasets to be processed.
  - **Parameters**:
    - *path*: Path to the dataset or folder.
    - *batch_mode*: If batch mode is used or only one dataset.
  - **Return**:  A list of the path to datasets.
- ***convert_xes_to_csv(xes_path: str, use_act: bool = True, use_time: bool = True,
use_res: bool = True, other_cols: dict = None, csv_path: str = None) -> str:***
Convert the XES file with the dataset to a CSV format file.
  - **Parameters**:
    - *xes_path*: Full path to the XES file.
    - *use_act*: Boolean indicating if use activity column.
    - *use_time*: Boolean indicating if use timestamp column.
    - *use_res*: Boolean indicating if use resource column.
    - *csv_path*: Path where the csv file will be stored
  - **Return**: Full path to the CSV file.

### embedding_generator
It contains the functions to train the different embedding models and 
to retrieve the generated embeddings.

### prediction_models

