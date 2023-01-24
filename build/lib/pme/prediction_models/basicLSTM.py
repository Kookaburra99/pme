import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from time import time

from ..utils import get_device


class LSTM_onehot(nn.Module):
    """
    Implementation of a LSTM with onehot inputs for next activity prediction
    """

    def __init__(self, num_categories: int, device: torch.device, seed: int = 21):
        """
        Creates the model for next activity prediction with input activities as onehots
        :param num_categories: Number of unique categories (onehot size)
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.num_categories = num_categories

        self.lstm = nn.LSTM(input_size=num_categories,
                            hidden_size=100,
                            batch_first=True)
        self.batchnorm = nn.BatchNorm1d(100)
        self.act_output = nn.Linear(in_features=100,
                                    out_features=num_categories)

        self.to(device)

    def forward(self, prefixes):
        x, _ = self.lstm(prefixes)
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        x = x[:, :, -1]
        act_output = self.act_output(x)

        return act_output

    def get_onehotprefixes_loader(self, cases: list[list],
                                  batch_size: int, split: str):
        """
        Generates prefixes with a corresponding activity
        :param cases: List of lists, each of which contains the activities of each case
        :param batch_size: Size of the mini-batches
        :param split: Partition to use (train, val or test)
        """
        prefixes = []
        next_acts = []

        for case in cases:
            for i in range(len(case)):
                # This would be an empty prefix: impossible to predict
                if i == 0:
                    continue
                next_acts.append(case[i])

                raw_prefix = case[0:i]
                prefix = []
                for act in raw_prefix:
                    onehot_act = torch.zeros(self.num_categories)
                    onehot_act[act] = 1
                    prefix.append(onehot_act)
                prefix = torch.stack(prefix)
                prefixes.append(prefix)

        prefixes = pad_sequence(prefixes, batch_first=True).to(self.device)
        next_acts = torch.LongTensor(next_acts).to(self.device)

        dataset = TensorDataset(prefixes, next_acts)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if split == 'train':
            self.train_loader = loader
        if split == 'val':
            self.val_loader = loader
        if split == 'test':
            self.test_loader = loader

    def save_model(self, path_model: str):
        """
        Save the full LSTM_onehot module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_model(self, learning_rate: float, epochs: int, early_stop: int = None,
                    path_model: str = './models/LSTM_onehot.m'):
        """
        Train the LSTM prediction model model with the stored 'train_loader'
        and validate it with the stored 'val_loader'. The best
        model in validation is kept
        :param learning_rate: The initial learning rate
        :param epochs: Number of epochs of training
        :param path_model: Full path where model will be stored.
        Default is './models/LSTM_onehot.m'
        :param early_stop: Number of epochs after which early stopping
        is performed if the validation loss does not improve
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        best_val_loss = np.inf
        trigger_times = 0

        for e in range(epochs):
            # Training
            self.train()
            for mini_batch in self.train_loader:
                prefix = mini_batch[0]
                target = mini_batch[1]

                self.zero_grad()
                pred_act = self(prefix)
                loss = loss_fn(pred_act, target)
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()

                val_sum_loss = []
                for mini_batch in self.val_loader:
                    prefix = mini_batch[0]
                    target = mini_batch[1]

                    pred_act = self(prefix)
                    loss = loss_fn(pred_act, target)
                    val_sum_loss.append(loss.item())

                val_avg_loss = np.mean(np.array(val_sum_loss))

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    trigger_times = 0
                    # Save new best model
                    self.save_model(path_model)
                else:
                    trigger_times += 1
                    # Early stopping
                    if early_stop and trigger_times == early_stop:
                        break

        self.load_best_model(path_model)

    def test_predictions(self) -> (list, list):
        """
        Make predictions based on the 'test_loader' and return
        the accuracy
        :return: The accuracy of the predictions in range (0, 1)
        """
        self.eval()

        list_correct_preds = []
        for mini_batch in self.test_loader:
            prefix = mini_batch[0]
            target = mini_batch[1]

            pred_act = self(prefix)
            pred_indices = torch.max(pred_act, 1).indices

            correct_predictions = (pred_indices == target).tolist()
            list_correct_preds.extend(correct_predictions)

        acc = np.array(list_correct_preds).mean()
        return acc


def train_test_LSTMonehot(train_cases: list[list], val_cases: list[list], test_cases: list[list],
                          num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
                          batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (float, float, float):
    """
    Train and test LSTM_onehot next activity prediction model
    :param train_cases: List of lists, each of which contains the activities of each case in training partition
    :param val_cases: List of lists, each of which contains the activities of each case in validation partition
    :param test_cases: List of lists, each of which contains the activities of each case in testing partition
    :param num_categories: Number of unique activities
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :param use_gpu: Boolean indicating if GPU for the training the model
    :return: The accuracy in test partition, the training time and the testing time
    """
    device = get_device(use_gpu)

    start_time = time()
    pred_model = LSTM_onehot(num_categories, device, seed)
    pred_model.get_onehotprefixes_loader(train_cases, batch_size, 'train')
    pred_model.get_onehotprefixes_loader(val_cases, batch_size, 'val')
    pred_model.get_onehotprefixes_loader(test_cases, batch_size, 'test')
    pred_model.train_model(learning_rate, epochs, 10)
    end_time = time()
    train_time = end_time - start_time
    print(f'Time to train LSTM onehot: {train_time}')

    start_time = time()
    acc = pred_model.test_predictions()
    end_time = time()
    test_time = end_time - start_time
    print(f'Time to test LSTM onehot: {test_time}')
    print(f'LSTM onehot accuracy: {acc}')

    return acc, train_time, test_time


########################################################################################################################

class LSTM_emblayer(nn.Module):
    """
    Implementation of a LSTM with embedding layer for next activity prediction
    """

    def __init__(self, num_categories: int, emb_size: int,
                 device: torch.device, seed: int = 21):
        """
        Creates the model for next activity prediction with embedding layer in the input
        :param num_categories: Number of unique categories (embedding matrix size)
        :param emb_size: Size of the embeddings
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.num_categories = num_categories

        self.embeddings = nn.Embedding(num_embeddings=num_categories + 1,  # Embedding padding representation
                                       embedding_dim=emb_size,
                                       padding_idx=num_categories)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=100,
                            batch_first=True)
        self.batchnorm = nn.BatchNorm1d(100)
        self.act_output = nn.Linear(in_features=100,
                                    out_features=num_categories)

        self.to(device)

    def forward(self, prefixes):
        embs = self.embeddings(prefixes)
        x, _ = self.lstm(embs)
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        x = x[:, :, -1]
        act_output = self.act_output(x)

        return act_output

    def get_labelprefixes_loader(self, cases: list[list],
                                 batch_size: int, split: str):
        """
        Generates prefixes with a corresponding activity
        :param cases: List of lists, each of which contains the activities of each case
        :param batch_size: Size of the mini-batches
        :param split: Partition to use (train, val or test)
        """
        prefixes = []
        next_acts = []

        for case in cases:
            for i in range(len(case)):
                # This would be an empty prefix: impossible to predict
                if i == 0:
                    continue
                next_acts.append(case[i])

                prefix = torch.LongTensor(case[0:i])
                prefixes.append(prefix)

        prefixes = pad_sequence(prefixes, batch_first=True,
                                padding_value=self.num_categories).to(self.device)
        next_acts = torch.LongTensor(next_acts).to(self.device)

        dataset = TensorDataset(prefixes, next_acts)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if split == 'train':
            self.train_loader = loader
        if split == 'val':
            self.val_loader = loader
        if split == 'test':
            self.test_loader = loader

    def save_model(self, path_model: str):
        """
        Save the full LSTM_emblayer module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_model(self, learning_rate: float, epochs: int, early_stop: int = None,
                    path_model: str = './models/LSTM_emblayer.m'):
        """
        Train the LSTM prediction model model with the stored 'train_loader'
        and validate it with the stored 'val_loader'. The best
        model in validation is kept
        :param learning_rate: The initial learning rate
        :param epochs: Number of epochs of training
        :param path_model: Full path where model will be stored.
        Default is './models/LSTM_onehot.m'
        :param early_stop: Number of epochs after which early stopping
        is performed if the validation loss does not improve
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        best_val_loss = np.inf
        trigger_times = 0

        for e in range(epochs):
            # Training
            self.train()
            for mini_batch in self.train_loader:
                prefix = mini_batch[0]
                target = mini_batch[1]

                self.zero_grad()
                pred_act = self(prefix)
                loss = loss_fn(pred_act, target)
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()

                val_sum_loss = []
                for mini_batch in self.val_loader:
                    prefix = mini_batch[0]
                    target = mini_batch[1]

                    pred_act = self(prefix)
                    loss = loss_fn(pred_act, target)
                    val_sum_loss.append(loss.item())

                val_avg_loss = np.mean(np.array(val_sum_loss))

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    trigger_times = 0
                    # Save new best model
                    self.save_model(path_model)
                else:
                    trigger_times += 1
                    # Early stopping
                    if early_stop and trigger_times == early_stop:
                        break

        self.load_best_model(path_model)

    def test_predictions(self) -> (list, list):
        """
        Make predictions based on the 'test_loader' and return
        the accuracy
        :return: The accuracy of the predictions in range (0, 1)
        """
        self.eval()

        list_correct_preds = []
        for mini_batch in self.test_loader:
            prefix = mini_batch[0]
            target = mini_batch[1]

            pred_act = self(prefix)
            pred_indices = torch.max(pred_act, 1).indices

            correct_predictions = (pred_indices == target).tolist()
            list_correct_preds.extend(correct_predictions)

        acc = np.array(list_correct_preds).mean()
        return acc


def train_test_LSTMemblayer(train_cases: list[list], val_cases: list[list], test_cases: list[list],
                            num_categories: int, emb_size: int, learning_rate: float = 0.05,
                            epochs: int = 200, batch_size: int = 32, seed: int = 21,
                            use_gpu: bool = True) -> (float, float, float):
    """
    Train and test LSTM_emblayer next activity prediction model
    :param train_cases: List of lists, each of which contains the activities of each case in training partition
    :param val_cases: List of lists, each of which contains the activities of each case in validation partition
    :param test_cases: List of lists, each of which contains the activities of each case in testing partition
    :param num_categories: Number of unique activities
    :param emb_size: Size of the embeddings
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :param use_gpu: Boolean indicating if GPU for the training the model
    :return: The accuracy in test partition, the training time and the testing time
    """
    device = get_device(use_gpu)

    start_time = time()
    pred_model = LSTM_emblayer(num_categories, emb_size, device, seed)
    pred_model.get_labelprefixes_loader(train_cases, batch_size, 'train')
    pred_model.get_labelprefixes_loader(val_cases, batch_size, 'val')
    pred_model.get_labelprefixes_loader(test_cases, batch_size, 'test')
    pred_model.train_model(learning_rate, epochs, 10)
    end_time = time()
    train_time = end_time - start_time
    print(f'Time to train LSTM emblayer: {train_time}')

    start_time = time()
    acc = pred_model.test_predictions()
    end_time = time()
    test_time = end_time - start_time
    print(f'Time to test LSTM emblayer: {test_time}')
    print(f'LSTM emblayer accuracy: {acc}')

    return acc, train_time, test_time


########################################################################################################################

class LSTM_embeddings(nn.Module):
    """
    Implementation of a LSTM with pre-trained embeddings as inputs for next activity prediction
    """

    def __init__(self, num_categories: int, embeddings_dict: dict,
                 device: torch.device, seed: int = 21):
        """
        Creates the model for next activity prediction with input activities as
        pre-trained embeddings
        :param num_categories: Number of unique categories (onehot size)
        :param embeddings_dict: Dictionary with the activities and their embeddings
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.num_categories = num_categories
        self.embeddings_dict = embeddings_dict
        self.emb_size = len(list(embeddings_dict.values())[0])

        self.lstm = nn.LSTM(input_size=self.emb_size,
                            hidden_size=100,
                            batch_first=True)
        self.batchnorm = nn.BatchNorm1d(100)
        self.act_output = nn.Linear(in_features=100,
                                    out_features=num_categories)

        self.to(device)

    def forward(self, prefixes):
        x, _ = self.lstm(prefixes)
        x = torch.permute(x, (0, 2, 1))
        x = self.batchnorm(x)
        x = x[:, :, -1]
        act_output = self.act_output(x)

        return act_output

    def get_embprefixes_loader(self, cases: list[list],
                               batch_size: int, split: str):
        """
        Generates prefixes with a corresponding activity
        :param cases: List of lists, each of which contains the activities of each case
        :param batch_size: Size of the mini-batches
        :param split: Partition to use (train, val or test)
        """
        prefixes = []
        next_acts = []

        for case in cases:
            for i in range(len(case)):
                # This would be an empty prefix: impossible to predict
                if i == 0:
                    continue
                next_acts.append(case[i])

                raw_prefix = case[0:i]
                prefix = []
                for act in raw_prefix:
                    emb_act = self.embeddings_dict[act]
                    prefix.append(emb_act)

                prefix = torch.Tensor(np.array(prefix))
                prefixes.append(prefix)

        prefixes = pad_sequence(prefixes, batch_first=True).to(self.device)
        next_acts = torch.LongTensor(next_acts).to(self.device)

        dataset = TensorDataset(prefixes, next_acts)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if split == 'train':
            self.train_loader = loader
        if split == 'val':
            self.val_loader = loader
        if split == 'test':
            self.test_loader = loader

    def save_model(self, path_model: str):
        """
        Save the full LSTM_embeddings module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_model(self, learning_rate: float, epochs: int, early_stop: int = None,
                    path_model: str = './models/LSTM_embeddings.m'):
        """
        Train the LSTM prediction model model with the stored 'train_loader'
        and validate it with the stored 'val_loader'. The best
        model in validation is kept
        :param learning_rate: The initial learning rate
        :param epochs: Number of epochs of training
        :param path_model: Full path where model will be stored.
        Default is './models/LSTM_onehot.m'
        :param early_stop: Number of epochs after which early stopping
        is performed if the validation loss does not improve
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        best_val_loss = np.inf
        trigger_times = 0

        for e in range(epochs):
            # Training
            self.train()
            for mini_batch in self.train_loader:
                prefix = mini_batch[0]
                target = mini_batch[1]

                self.zero_grad()
                pred_act = self(prefix)
                loss = loss_fn(pred_act, target)
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()

                val_sum_loss = []
                for mini_batch in self.val_loader:
                    prefix = mini_batch[0]
                    target = mini_batch[1]

                    pred_act = self(prefix)
                    loss = loss_fn(pred_act, target)
                    val_sum_loss.append(loss.item())

                val_avg_loss = np.mean(np.array(val_sum_loss))

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    trigger_times = 0
                    # Save new best model
                    self.save_model(path_model)
                else:
                    trigger_times += 1
                    # Early stopping
                    if early_stop and trigger_times == early_stop:
                        break

        self.load_best_model(path_model)

    def test_predictions(self) -> (list, list):
        """
        Make predictions based on the 'test_loader' and return
        the accuracy
        :return: The accuracy of the predictions in range (0, 1)
        """
        self.eval()

        list_correct_preds = []
        for mini_batch in self.test_loader:
            prefix = mini_batch[0]
            target = mini_batch[1]

            pred_act = self(prefix)
            pred_indices = torch.max(pred_act, 1).indices

            correct_predictions = (pred_indices == target).tolist()
            list_correct_preds.extend(correct_predictions)

        acc = np.array(list_correct_preds).mean()
        return acc


def train_test_LSTMembeddings(train_cases: list[list], val_cases: list[list], test_cases: list[list],
                              num_categories: int, embeddings_dict: dict, learning_rate: float = 0.05,
                              epochs: int = 200, batch_size: int = 32, seed: int = 21,
                              use_gpu: bool = True) -> (float, float, float):
    """
    Train and test LSTM_embeddings next activity prediction model
    :param train_cases: List of lists, each of which contains the activities of each case in training partition
    :param val_cases: List of lists, each of which contains the activities of each case in validation partition
    :param test_cases: List of lists, each of which contains the activities of each case in testing partition
    :param num_categories: Number of unique activities
    :param embeddings_dict: Dictionary with the activities and their embeddings
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :param use_gpu: Boolean indicating if GPU for the training the model
    :return: The accuracy in test partition, the training time and the testing time
    """
    device = get_device(use_gpu)

    start_time = time()
    pred_model = LSTM_embeddings(num_categories, embeddings_dict, device, seed)
    pred_model.get_embprefixes_loader(train_cases, batch_size, 'train')
    pred_model.get_embprefixes_loader(val_cases, batch_size, 'val')
    pred_model.get_embprefixes_loader(test_cases, batch_size, 'test')
    pred_model.train_model(learning_rate, epochs, 10)
    end_time = time()
    train_time = end_time - start_time
    print(f'Time to train LSTM embeddings: {train_time}')

    start_time = time()
    acc = pred_model.test_predictions()
    end_time = time()
    test_time = end_time - start_time
    print(f'Time to test LSTM embeddings: {test_time}')
    print(f'LSTM embeddings accuracy: {acc}')

    return acc, train_time, test_time
