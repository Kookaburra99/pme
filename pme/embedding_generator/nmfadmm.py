import networkx as nx
from time import time
from karateclub import NMFADMM


def get_nmfadmm_embeddings(graph: nx.Graph, emb_size: int,
                           epochs: int = 200, seed: int = 21) -> (dict, float):
    """
    Train NMF-ADMM graph embeddings and return
    a dictionary with pairs [activity identifier - embedding]
    :param graph: Networkx Graph with the structure of the process
    :param emb_size: Size of the embeddings generated
    :param epochs: Number of epochs of training
    :param seed: Seed to set the random state and get reproducibility
    :return: Dictionary with the embeddings and the time expended during the training
    """

    start_time = time()
    nmfadmm = NMFADMM(dimensions=emb_size, iterations=epochs, seed=seed)
    nmfadmm.fit(graph)
    end_time = time()

    dict_embeddings = {}
    np_embeddings = nmfadmm.get_embedding()
    for i in range(len(np_embeddings)):
        dict_embeddings.update({
            i: np_embeddings[i]
        })

    return dict_embeddings, end_time - start_time
