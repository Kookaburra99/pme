import networkx as nx
from time import time
from karateclub import Walklets


def get_walklets_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, walk_number: int = 10,
                            walk_length: int = 10, seed: int = 21) -> (dict, float):
    """
    Train Walklets graph embeddings and return
    a dictionary with pairs [activity identifier - embedding]
    :param graph: Networkx Graph with the structure of the process
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param walk_number: Number of random walks from each node
    :param walk_length: Length of each random walk
    :param seed: Seed to set the random state and get reproducibility
    :return: Dictionary with the embeddings and the time expended during the training
    """
    start_time = time()
    walklets = Walklets(walk_number=walk_number, walk_length=walk_length,
                        dimensions=emb_size, workers=1, window_size=win_size,
                        epochs=epochs, learning_rate=learning_rate, min_count=1, seed=seed)
    walklets.fit(graph)
    end_time = time()

    dict_embeddings = {}
    np_embeddings = walklets.get_embedding()
    for i in range(len(np_embeddings)):
        dict_embeddings.update({
            i: np_embeddings[i]
        })

    return dict_embeddings, end_time - start_time
