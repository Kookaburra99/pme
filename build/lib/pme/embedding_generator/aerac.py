import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from time import time

from ..utils import get_device


class AErac(nn.Module):
    """
    Implementation of AErac (AutoEncoder to Reconstruct
    the Context of an Activity) model
    """

    def __init__(self, num_categories: int, win_size: int,
                 emb_size: int, device: torch.device, seed: int = 21):
        """
        Creates the model for AErac embeddings
        :param num_categories: Number of unique categories (number
        of different embeddings)
        :param win_size: Size of the context window
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
                                                   bias=False) for i in range(2 * win_size + 1)])

        self.hidden_layer_in = nn.Linear(in_features=(2 * win_size + 1) * emb_size,
                                         out_features=emb_size,
                                         bias=False)

        self.hidden_layer_out = nn.Linear(in_features=emb_size,
                                          out_features=(2 * win_size + 1) * emb_size,
                                          bias=False)

        self.output_acts = nn.ModuleList([nn.Linear(in_features=(2 * win_size + 1) * emb_size,
                                                    out_features=num_categories,
                                                    bias=False) for i in range(2 * win_size + 1)])

        self.to(device)

    def encode(self, x: torch.Tensor):
        h = torch.Tensor().to(self.device)
        for i in range(2 * self.win_size + 1):
            x_split = x[:, i * self.num_categories:(i + 1) * self.num_categories]
            h_split = F.relu(self.input_acts[i](x_split))
            h = torch.cat([h, h_split], dim=-1)
        h = F.relu(self.hidden_layer_in(h))
        return h

    def decode(self, x: torch.Tensor):
        h = F.relu(self.hidden_layer_out(x))
        out = []
        for i in range(2 * self.win_size + 1):
            out_split = self.output_acts[i](h)
            out.append(out_split)
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
                context = torch.zeros((2*self.win_size+1) * self.num_categories,
                                      device=self.device)
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
        Save the full AErac module
        :param path_model: Full path where model will be stored
        """
        directory = pathlib.Path(path_model).parent
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        torch.save(self, path_model)

    def load_best_model(self, path_model: str):
        self = torch.load(path_model)

    def train_embeddings(self, learning_rate: float, epochs: int, early_stop: int = None,
                         path_model: str = './models/AErac_model.m'):
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

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        best_val_loss = np.inf
        trigger_times = 0

        self.train()
        for e in range(epochs):
            # Training
            self.train()

            for mini_batch in self.train_loader:
                inputs = mini_batch[0]
                outputs = mini_batch[1]
                outputs = torch.split(outputs,
                                      [self.num_categories for _
                                       in range(2 * self.win_size + 1)], dim=-1)

                self.zero_grad()
                reconstruction = self(inputs)
                loss = 0
                for i in range(2 * self.win_size + 1):
                    loss = loss + loss_fn(reconstruction[i], outputs[i])
                loss.backward()
                optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()

                val_sum_loss = []
                for mini_batch in self.val_loader:
                    inputs = mini_batch[0]
                    outputs = mini_batch[1]
                    outputs = torch.split(outputs,
                                          [self.num_categories for _
                                           in range(2 * self.win_size + 1)], dim=-1)

                    reconstruction = self(inputs)
                    loss = 0
                    for i in range(2 * self.win_size + 1):
                        loss = loss + loss_fn(reconstruction[i], outputs[i])
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

    def get_reconstruction_accuracy(self, split: str) -> float:
        """
        Test the accuracy of the autoencoder reconstructing the context
        :param split: Partition to use (train, val or test)
        :return: The accuracy reconstructing contexts
        """
        if split == 'train':
            loader = self.train_loader
        if split == 'val':
            loader = self.val_loader
        if split == 'test':
            loader = self.test_loader

        self.eval()

        val_sum_acc = []
        for mini_batch in loader:
            inputs = mini_batch[0]
            outputs = mini_batch[1]
            outputs = torch.split(outputs, [self.num_categories for _
                                            in range(2 * self.win_size + 1)], dim=-1)

            reconstruction = self(inputs)
            for i in range(2 * self.win_size + 1):
                idx_outputs = torch.argmax(outputs[i], dim=-1)
                idx_reconst = torch.argmax(reconstruction[i], dim=-1)
                val_sum_acc.extend(np.array((idx_outputs == idx_reconst).cpu()))

        return np.mean(val_sum_acc).item()


def get_aerac_embeddings(train_cases: list[list], val_cases: list[list], win_size: int, emb_size: int,
                        num_categories: int, learning_rate: float = 0.05, epochs: int = 200,
                        batch_size: int = 32, seed: int = 21, use_gpu: bool = True) -> (dict, float, float):
    """
    Train AErac embeddings and return a dictionary with pairs [activity identifier - embedding]
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
    and the reconstruction accuracy in the validation split
    """
    device = get_device(use_gpu)

    start_time = time()
    aerac_model = AErac(num_categories, win_size, emb_size, device, seed)
    aerac_model.get_autoencoder_dataloader(train_cases, batch_size, 'train')
    aerac_model.get_autoencoder_dataloader(val_cases, batch_size, 'val')
    aerac_model.train_embeddings(learning_rate, epochs, 10)
    end_time = time()

    accuracy = aerac_model.get_reconstruction_accuracy('val')

    dict_embeddings = {}
    weights = list(aerac_model.input_acts[win_size].parameters())[0].data
    for i in range(aerac_model.num_categories):
        dict_embeddings.update({
            i: weights[:, i].cpu().numpy()
        })

    return dict_embeddings, end_time - start_time, accuracy
