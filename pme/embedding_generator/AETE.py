import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from time import time

from ..utils import get_device


class AETE(nn.Module):
    """
    Implementation of AETE (AutoEncoder to Train Embeddings) model
    """

    def __init__(self, num_categories: int, win_size: int,
                 emb_size: int, device: torch.device, seed: int = 21):
        """
        Creates the model for AETE embeddings
        :param num_categories: Number of unique categories (number
        of different embeddings)
        :param win_size: Size of the context windows
        :param emb_size: Size of the embedding
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        """

        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.num_categories = num_categories
        self.emb_size = emb_size
        self.win_size = win_size

        self.input_acts = nn.ModuleList([nn.Linear(in_features=num_categories,
                                                   out_features=emb_size,
                                                   bias=False) for i in range(2*win_size+1)])

        self.hidden_layer_in = nn.Linear(in_features=(2*win_size+1) * emb_size,
                                         out_features=emb_size,
                                         bias=False)
        self.hidden_layer_out = nn.Linear(in_features=emb_size,
                                          out_features=(2*win_size+1) * emb_size,
                                          bias=False)

        self.output_acts = nn.ModuleList([nn.Linear(in_features=emb_size,
                                                    out_features=num_categories,
                                                    bias=False) for i in range(2*win_size+1)])

        self.to(device)

    def encode(self, x: torch.Tensor):
        h = torch.Tensor()
        for i in range(2 * self.win_size + 1):
            x_split = x[:, i * self.num_categories:(i + 1) * self.num_categories]
            h_split = F.relu(self.input_acts[i](x_split))
            h = torch.cat([h, h_split], dim=-1)
        h = F.relu(self.hidden_layer_in(h))
        return h

    def decode(self, x: torch.Tensor):
        h = F.relu(self.hidden_layer_out(x))
        out = torch.Tensor()
        for i in range(2 * self.win_size + 1):
            h_split = h[:, i * self.emb_size:(i + 1) * self.emb_size]
            out_split = F.softmax(self.output_acts[i](h_split), dim=-1)
            out = torch.cat([out, out_split], dim=-1)
        return out

    def forward(self, inputs):
        x = self.encode(inputs)
        output = self.decode(x)
        return output

    def get_autoencoder_dataloader(self, cases: list[list],
                                   batch_size: int, split: str):
        """
        Get each feature from every case and their context represented
        in one vector
        :param cases: List of lists, each of which contains the activities of each case
        :param batch_size: Size of the mini-batches
        :param split: Partition to use (train, val or test)
        """
        contexts = []

        for case in cases:
            for i in range(len(case)):
                context = torch.zeros((2*self.win_size+1) * self.num_categories)
                context[self.win_size*self.num_categories + case[i]] = 1

                for w in range(self.win_size):
                    # Context behind
                    if i - 1 - w >= 0:
                        pos_onehot = self.win_size - 1 - w
                        context[(pos_onehot*self.num_categories) + case[i - 1 - w]] = 1
                    # Context ahead
                    if i + 1 + w < len(case):
                        pos_onehot = self.win_size + 1 + w
                        context[(pos_onehot*self.num_categories) + case[i + 1 + w]] = 1

                contexts.append(context)

        contexts = torch.stack(contexts).to(self.device)

        dataset = TensorDataset(contexts, contexts)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if split == 'train':
            self.train_loader = loader
        if split == 'val':
            self.val_loader = loader
        if split == 'test':
            self.test_loader = loader

    def save_model(self, path_model: str):
        """
        Save the full AETE module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_embeddings(self, learning_rate: float, epochs: int, early_stop: int = None,
                         path_model: str = './models/AETE_model.m'):
        """
        Train the AETE model with the stored 'train_loader'
        and validate it with the stored 'val_loader'. The best
        model in validation is kept
        :param learning_rate: The initial learning rate
        :param epochs: Number of epochs of training
        :param path_model: Full path where model will be stored.
        Default is './models/AETE_model.m'
        :param early_stop: Number of epochs after which early stopping
        is performed if the validation loss does not improve
        """

        #ToDo Revisar que función de pérdida usar: probablemente MAE o MSE

