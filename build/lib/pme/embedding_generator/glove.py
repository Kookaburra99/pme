import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from time import time

from ..utils import get_device


class GloVe(nn.Module):
    """
    Implementation of GloVe model from NLP to Process Mining
    """

    def __init__(self, vocab_size: int, emb_size: int, device: torch.device,
                 seed: int = 21, x_max: int = 100, alpha: float = 3 / 4):
        """
        Creates the model for GloVe embeddings
        :param vocab_size: Number of unique categories (number of different
        embeddings)
        :param emb_size: Size of the embedding
        :param device: CPU or GPU, where the model will be executed
        :param seed: Seed to set the random state and get reproducibility
        :param x_max:
        :param alpha:
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = device
        self.vocab_size = vocab_size

        self.alpha = alpha
        self.x_max = x_max

        self.focal_embeddings = nn.Embedding(
            vocab_size, emb_size)
        self.context_embeddings = nn.Embedding(
            vocab_size, emb_size)
        self.focal_biases = nn.Parameter(
            torch.randn(vocab_size, dtype=torch.float))
        self.context_biases = nn.Parameter(
            torch.randn(vocab_size, dtype=torch.float))

        self.to(device)

    def forward(self, focal_input, context_input, cooc_count):
        focal_emb = self.focal_embeddings(focal_input)
        context_emb = self.context_embeddings(context_input)
        focal_bias = self.focal_biases[focal_input]
        context_bias = self.context_biases[context_input]

        emb_prods = torch.mul(focal_emb, context_emb).sum(dim=1)
        distance_expr = (emb_prods + focal_bias + context_bias - cooc_count.log()).square()

        weight_factor = (cooc_count / self.x_max).float_power(self.alpha).clamp(0, 1)
        loss = torch.mul(weight_factor, distance_expr).mean()
        return loss

    def get_coocurrence_dataloader(self, cases: list[list], win_size: int,
                                   batch_size: int):
        """
        Get each feature and the distance with each other feature in its context.
        The distance is weighted by the window size. Generate a Dataloader and store
        it in the 'train_loader' attribute of the class
        :param cases: List of lists, each of which contains the activities of each case
        :param win_size: Size of the context window
        :param batch_size: Size of the mini-batches
        """
        cooc_matrix = defaultdict(float)

        for case in cases:
            for i in range(len(case)):
                current = case[i]

                for w in range(win_size):
                    distance = w + 1

                    # Context behind
                    if i - 1 - w >= 0:
                        context = case[i - 1 - w]
                        cooc_matrix[(current, context)] += 1 / distance
                    # Context ahead
                    if i + 1 + w < len(case):
                        context = case[i + 1 + w]
                        cooc_matrix[(current, context)] += 1 / distance

        current_acts = []
        context_acts = []
        counts = []
        for (current, ctx), count in cooc_matrix.items():
            current_acts.append(torch.LongTensor([current]))
            context_acts.append(torch.LongTensor([ctx]))
            counts.append(torch.Tensor([count]))

        current_acts = torch.stack(current_acts).to(self.device)
        context_acts = torch.stack(context_acts).to(self.device)
        counts = torch.stack(counts).to(self.device)

        dataset = TensorDataset(current_acts, context_acts, counts)
        self.train_loader = DataLoader(dataset, batch_size)

    def train_embeddings(self, learning_rate: float, epochs: int):
        """
        Train the GloVe model with the stored 'train_loader'
        param learning_rate: The initial learning rate
        :param learning_rate: The learning rate of the training
        :param epochs: Number of epochs of training
        """

        optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)

        self.train()
        for e in range(epochs):
            for mini_batch in self.train_loader:
                focal_act, ctx_act, counts = mini_batch

                self.zero_grad()
                loss = self(focal_act, ctx_act, counts)
                loss.backward()
                optimizer.step()
        self.eval()


def get_glove_embeddings(cases: list[list], win_size: int, emb_size: int, vocab_size: int,
                         learning_rate: float = 0.05, epochs: int = 200, batch_size: int = 32,
                         seed: int = 21, use_gpu: bool = True) -> (dict, float):
    """
    Train GloVe embeddings and return a dictionary with pairs [activity identifier - embedding]
    :param cases: List of lists, each of which contains the activities of each case
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param vocab_size: Number of categories (embeddings generated)
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :param use_gpu: Boolean indicating if GPU for the training of the embeddings
    :return: Dictionary with the embeddings and the time expended during the training
    """

    device = get_device(use_gpu)

    start_time = time()
    glove_model = GloVe(vocab_size, emb_size, device, seed)
    glove_model.get_coocurrence_dataloader(cases, win_size, batch_size)
    glove_model.train_embeddings(learning_rate, epochs)
    end_time = time()

    dict_embeddings = {}
    embedding_layer = glove_model.focal_embeddings.cpu()
    for i in range(glove_model.vocab_size):
        dict_embeddings.update({
            i: embedding_layer(torch.LongTensor([i])).squeeze().detach().numpy()
        })

    return dict_embeddings, end_time - start_time
