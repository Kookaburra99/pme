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
- ***get_datasets_list(path: str, batch_mode: bool) -> list***:
Get list of paths to datasets to be processed.
  - **Parameters**:
    - *path*: Path to the dataset or folder.
    - *batch_mode*: If batch mode is used or only one dataset.
  - **Return**:  A list of the path to datasets.
- ***convert_xes_to_csv(xes_path: str, use_act: bool = True, use_time: bool = True,
use_res: bool = True, csv_path: str = None) -> str:***
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

#### word2vec
- ***get_skipgram_embeddings(cases: list[list], win_size: int, emb_size: int,
                            learning_rate: float = 0.002, min_lr: float = 0.002,
                            ns_rate: int = 0, epochs: int = 200, batch_size: int = 32,
                            seed: int = 21) -> (dict, float):*** 
Train Word2Vec embeddings using Skipgram methods and return
a dictionary with pairs [activity identifier - embedding]
  - **Parameters**:
    - *cases*: List of lists, each of which contains the activities of each case.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *min_lr*: Learning rate will linearly drop to this value as training progresses.
    - *ns_rate*: Integer indicating the ratio of negative samples for each positive sample.
    If 0, no negative sampling is used.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.
- ***get_cbow_embeddings(cases: list[list], win_size: int, emb_size: int,
                            learning_rate: float = 0.002, min_lr: float = 0.002,
                            ns_rate: int = 0, epochs: int = 200, batch_size: int = 32,
                            seed: int = 21) -> (dict, float):*** 
Train Word2Vec embeddings using CBOW methods and return
a dictionary with pairs [activity identifier - embedding]
  - **Parameters**:
    - *cases*: List of lists, each of which contains the activities of each case.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *min_lr*: Learning rate will linearly drop to this value as training progresses.
    - *ns_rate*: Integer indicating the ratio of negative samples for each positive sample.
    If 0, no negative sampling is used.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### glove
- ***get_glove_embeddings(cases: list[list], win_size: int, emb_size: int, vocab_size: int,
                         learning_rate: float = 0.05, epochs: int = 200, batch_size: int = 32,
                         seed: int = 21, use_gpu: bool = True) -> (dict, float):***
Train GloVe embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *cases*: List of lists, each of which contains the activities of each case.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *vocab_size*: Number of categories (embeddings generated).
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training of the embeddings.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### acov
- ***get_acov_embeddings(train_cases: list[list], val_cases: list[list], win_size: int, emb_size: int,
                        num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
                        batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (dict, float):***
Train ACOV embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *train_cases*: List of lists, each of which contains the activities of each case in training partition.
    - *val_cases*: List of lists, each of which contains the activities of each case in validation partition.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *num_categories*: Number of unique elements (embeddings generated).
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training of the embeddings.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### dwc
- ***get_dwc_embeddings(train_cases: list[list], val_cases: list[list], win_size: int, emb_size: int,
                       num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
                       batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (dict, float):***
Train DWC embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *train_cases*: List of lists, each of which contains the activities of each case in training partition.
    - *val_cases*: List of lists, each of which contains the activities of each case in validation partition.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *num_categories*: Number of unique elements (embeddings generated).
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training of the embeddings.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### deepwalk
- ***get_deepwalk_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, walk_number: int = 10,
                            walk_length: int = 10, seed: int = 21) -> (dict, float):***
Train DeepWalk graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *walk_number*: Number of random walks from each node.
    - *walk_length*: Length of each random walk.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### node2vec
- ***get_node2vec_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
learning_rate: float = 0.002, epochs: int = 200, walk_number: int = 10,
walk_length: int = 10, p: float = 1.0, q: float = 1.0, seed: int = 21) -> (dict, float):***
Train DeepWalk graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *walk_number*: Number of random walks from each node.
    - *walk_length*: Length of each random walk.
    - *p*: Return parameter (1/p transition probability) to move towards from previous node.
    - *q*: In-out parameter (1/q transition probability) to move away from previous node.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### walklets
- ***get_walklets_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, walk_number: int = 10,
                            walk_length: int = 10, seed: int = 21) -> (dict, float):***
Train Walklets graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *walk_number*: Number of random walks from each node.
    - *walk_length*: Length of each random walk.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### role2vec
- *** get_role2vec_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, walk_number: int = 10,
                            walk_length: int = 10, seed: int = 21) -> (dict, float):***
Train Role2Vec graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *walk_number*: Number of random walks from each node.
    - *walk_length*: Length of each random walk.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### laplacianeigenmaps
- ***get_lapaclianeigenmaps_embeddings(graph: nx.Graph, emb_size: int,
                                      epochs: int = 200, seed: int = 21) -> (dict, float):***
Train Laplacian Eigenmpas graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *emb_size*: Size of the embeddings generated.
    - *epochs*: Number of epochs of training.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### diff2vec
- ***get_diff2vec_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, diffusion_number: int = 10,
                            diffusion_cover: int = 10, seed: int = 21) -> (dict, float):***
Train Diff2Vec graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *win_size*: Size of the window context.
    - *emb_size*: Size of the embeddings generated.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *diffusion_number*: Number of diffusions.
    - *diffusion_cover*: Number of nodes in diffusion.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### glee
- ***get_glee_embeddings(graph: nx.Graph, emb_size: int,
                        seed: int = 21) -> (dict, float):***
Train GLEE graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *emb_size*: Size of the embeddings generated.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

#### nmfadmm
- ***get_nmfadmm_embeddings(graph: nx.Graph, emb_size: int,
                           epochs: int = 200, seed: int = 21) -> (dict, float):***
Train NMF-ADMM graph embeddings and return a dictionary with pairs [activity identifier - embedding].
  - **Parameters**:
    - *graph*: Networkx Graph with the structure of the process.
    - *emb_size*: Size of the embeddings generated.
    - *epochs*: Number of epochs of training.
    - *seed*: Seed to set the random state and get reproducibility.
  - **Return**: Dictionary with the embeddings and the time expended during the training.

### prediction_models

#### basicLSTM
- ***train_test_LSTMonehot(train_cases: list[list], val_cases: list[list], test_cases: list[list],
num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (float, float, float):***
Train and test LSTM_onehot next activity prediction model
  - **Parameters**:
    - *train_cases*: List of lists, each of which contains the activities of each case in training partition.
    - *val_cases*: List of lists, each of which contains the activities of each case in validation partition.
    - *test_cases*: List of lists, each of which contains the activities of each case in testing partition.
    - *num_categories*: Number of unique activities.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training the model.
  - **Return**: The accuracy in test partition, the training time and the testing time.

- ***train_test_LSTMemblayer(train_cases: list[list], val_cases: list[list], test_cases: list[list],
                            num_categories: int, emb_size: int, learning_rate: float = 0.05,
                            epochs: int = 200, batch_size: int = 32, seed: int = 21,
                            use_gpu: bool = True) -> (float, float, float):***
Train and test LSTM_emblayer next activity prediction model
  - **Parameters**:
    - *train_cases*: List of lists, each of which contains the activities of each case in training partition.
    - *val_cases*: List of lists, each of which contains the activities of each case in validation partition.
    - *test_cases*: List of lists, each of which contains the activities of each case in testing partition.
    - *num_categories*: Number of unique activities.
    - *emb_size*: Size of the embeddings.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training the model.
  - **Return**: The accuracy in test partition, the training time and the testing time.

- ***train_test_LSTMembeddings(train_cases: list[list], val_cases: list[list], test_cases: list[list],
                              num_categories: int, embeddings_dict: dict, learning_rate: float = 0.05,
                              epochs: int = 200, batch_size: int = 32, seed: int = 21,
                              use_gpu: bool = True) -> (float, float, float):***
Train and test LSTM_embeddings next activity prediction model
  - **Parameters**: 
    - *train_cases*: List of lists, each of which contains the activities of each case in training partition.
    - *val_cases*: List of lists, each of which contains the activities of each case in validation partition.
    - *test_cases*: List of lists, each of which contains the activities of each case in testing partition.
    - *num_categories*: Number of unique activities.
    - *embeddings_dict*: Dictionary with the activities and their embeddings.
    - *learning_rate*: The initial learning rate.
    - *epochs*: Number of epochs of training.
    - *batch_size*: Size of the mini-batches.
    - *seed*: Seed to set the random state and get reproducibility.
    - *use_gpu*: Boolean indicating if GPU for the training the model.
  - **Return**: The accuracy in test partition, the training time and the testing time.

