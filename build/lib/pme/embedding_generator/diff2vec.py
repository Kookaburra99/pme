import networkx as nx
from time import time
from karateclub import Diff2Vec


def get_diff2vec_embeddings(graph: nx.Graph, win_size: int, emb_size: int,
                            learning_rate: float = 0.002, epochs: int = 200, diffusion_number: int = 10,
                            diffusion_cover: int = 10, seed: int = 21) -> (dict, float):
    """
    Train Diff2Vec graph embeddings and return
    a dictionary with pairs [activity identifier - embedding]
    :param graph: Networkx Graph with the structure of the process
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param learning_rate: The initial learning rate
    :param epochs: Number of epochs of training
    :param diffusion_number: Number of diffusions
    :param diffusion_cover: Number of nodes in diffusion
    :param seed: Seed to set the random state and get reproducibility
    :return: Dictionary with the embeddings and the time expended during the training
    """

    start_time = time()
    diff2vec = Diff2Vec(diffusion_number=diffusion_number, diffusion_cover=diffusion_cover,
                        dimensions=emb_size, workers=1, window_size=win_size, learning_rate=learning_rate,
                        epochs=epochs, min_count=1, seed=seed)
    diff2vec.fit(graph)
    end_time = time()

    dict_embeddings = {}
    np_embeddings = diff2vec.get_embedding()
    for i in range(len(np_embeddings)):
        dict_embeddings.update({
            i: np_embeddings[i]
        })

    return dict_embeddings, end_time - start_time
