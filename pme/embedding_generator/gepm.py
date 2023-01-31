import pathlib
import networkx as nx
from time import time
import numpy as np
import torch
import torch.nn as nn

from ..utils import get_device


class GEPM(nn.Module):
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

        self.layer1 = nn.Linear()
        self.layer2 = nn.Linear()

        self.to(device)
