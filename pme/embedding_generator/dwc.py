import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from time import time

from ..utils import get_device


class DWC(nn.Module):
    """
    Implementation of DWC (Distance-Weighted-Context) model
    """

    def __init__(self, num_categories: int, emb_size: int,
                 device: torch.device, seed: int = 21):
        """
        Creates the model for DWC embeddings
        :param num_categories: Number of unique categories (number
        of different embeddings)
        :param emb_size: Size of the embedding
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.num_categories = num_categories

        self.layer1 = nn.Linear(in_features=2 * num_categories,
                                out_features=emb_size,
                                bias=False)
        self.layer2 = nn.Linear(in_features=emb_size,
                                out_features=num_categories)

        self.to(device)

    def forward(self, inputs):
        x = self.layer1(inputs)
        output = self.layer2(x)

        return output

    def get_distanceweightedcontext_loader(self, cases: list[list], win_size: int,
                                           batch_size: int, split: str):
        """
        Get each feature from every case and their context represented
        in one vector
        :param cases: List of lists, each of which contains the activities of each case
        :param win_size: Size of the context window
        :param batch_size: Size of the mini-batches
        :param split: Partition to use (train, val or test)
        """
        inputs = []
        outputs = []

        for case in cases:
            for i in range(len(case)):
                current_act = torch.zeros(self.num_categories)
                current_act[case[i]] = 1

                context_beh = torch.zeros(self.num_categories)
                context_ahe = torch.zeros(self.num_categories)

                for w in range(win_size):
                    weight = w + 1
                    # Context behind
                    if i - 1 - w >= 0:
                        context_beh[case[i - 1 - w]] += -1 / weight
                    # Context ahead
                    if i + 1 + w < len(case):
                        context_ahe[case[i + 1 + w]] += 1 / weight

                inputs.append(torch.cat([current_act, context_beh]))
                outputs.append(context_ahe)

                inputs.append(torch.cat([current_act, context_ahe]))
                outputs.append(context_beh)

        inputs = torch.stack(inputs).to(self.device)
        outputs = torch.stack(outputs).to(self.device)

        dataset = TensorDataset(inputs, outputs)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if split == 'train':
            self.train_loader = loader
        if split == 'val':
            self.val_loader = loader
        if split == 'test':
            self.test_loader = loader

    def save_model(self, path_model: str):
        """
        Save the full DWC module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_embeddings(self, learning_rate: float, epochs: int, early_stop: int = None,
                         path_model: str = './models/DWC_model.m'):
        """
        Train the DWC model with the stored 'train_loader'
        and validate it with the stored 'val_loader'. The best
        model in validation is kept
        :param learning_rate: The initial learning rate
        :param epochs: Number of epochs of training
        :param path_model: Full path where model will be stored.
        Default is './models/DWC_model.m'
        :param early_stop: Number of epochs after which early stopping
        is performed if the validation loss does not improve
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss().to(self.device)

        best_val_loss = np.inf
        trigger_times = 0

        for e in range(epochs):
            # Training
            self.train()
            for mini_batch in self.train_loader:
                current = mini_batch[0]
                context_real = mini_batch[1]

                self.zero_grad()
                context_pred = self(current)
                loss = loss_fn(context_pred, context_real)
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()

                val_sum_loss = []
                for mini_batch in self.val_loader:
                    current = mini_batch[0]
                    context_real = mini_batch[1]

                    context_pred = self(current)
                    loss = loss_fn(context_pred, context_real)
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


def get_dwc_embeddings(train_cases: list[list], val_cases: list[list], win_size: int, emb_size: int,
                       num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
                       batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (dict, float):
    """
    Train DWC embeddings and return a dictionary with pairs [activity identifier - embedding]
    :param train_cases: List of lists, each of which contains the activities of each case in training partition
    :param val_cases: List of lists, each of which contains the activities of each case in validation partition
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param num_categories: Number of unique elements (embeddings generated)
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :param use_gpu: Boolean indicating if GPU for the training of the embeddings
    :return: Dictionary with the embeddings and the time expended during the training
    """
    device = get_device(use_gpu)

    start_time = time()
    dwc_model = DWC(num_categories, emb_size, device, seed)
    dwc_model.get_distanceweightedcontext_loader(train_cases, win_size, batch_size, 'train')
    dwc_model.get_distanceweightedcontext_loader(val_cases, win_size, batch_size, 'val')
    dwc_model.train_embeddings(learning_rate, epochs, 10)
    end_time = time()

    dict_embeddings = {}
    weights = list(dwc_model.parameters())[0].data
    for i in range(dwc_model.num_categories):
        dict_embeddings.update({
            i: weights[:, i].cpu().numpy()
        })

    return dict_embeddings, end_time - start_time
