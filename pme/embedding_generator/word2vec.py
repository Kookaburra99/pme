from time import time
from gensim.models import Word2Vec


def get_skipgram_embeddings(cases: list[list], win_size: int, emb_size: int,
                            learning_rate: float = 0.002, min_lr: float = 0.002,
                            ns_rate: int = 0, epochs: int = 200, batch_size: int = 32,
                            seed: int = 21) -> (dict, float):
    """
    Train Word2Vec embeddings using skipgram methods and return
    a dictionary with pairs [activity identifier - embedding]
    :param cases: List of lists, each of which contains the activities of each case
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param learning_rate: The initial learning rate
    :param min_lr: Learning rate will linearly drop to this value as training progresses
    :param ns_rate: Integer indicating the ratio of negative samples for each positive sample.
    If 0, no negative sampling is used
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :return: Dictionary with the embeddings and the time expended during the training
    """
    start_time = time()
    w2v_model = Word2Vec(min_count=0, window=win_size, vector_size=emb_size,
                         alpha=learning_rate, min_alpha=min_lr,
                         negative=ns_rate, batch_words=batch_size, sg=1)
    w2v_model.build_vocab(cases)
    w2v_model.train(cases, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)
    end_time = time()

    dict_embeddings = {}
    for i in range(len(w2v_model.wv)):
        dict_embeddings.update({
            i: w2v_model.wv[i]
        })

    return dict_embeddings, end_time - start_time


def get_cbow_embeddings(cases: list[list], win_size: int, emb_size: int,
                        learning_rate: float = 0.002, min_lr: float = 0.002,
                        ns_rate: int = 0, epochs: int = 200, batch_size: int = 32,
                        seed: int = 21) -> (dict, float):
    """
    Train Word2Vec embeddings using CBOW methods and return
    a dictionary with pairs [activity identifier - embedding]
    :param cases: List of lists, each of which contains the activities of each case
    :param win_size: Size of the window context
    :param emb_size: Size of the embeddings generated
    :param learning_rate: The initial learning rate
    :param min_lr: Learning rate will linearly drop to this value as training progresses
    :param ns_rate: Integer indicating the ratio of negative samples for each positive sample.
    If 0, no negative sampling is used
    :param epochs: Number of epochs of training
    :param batch_size: Size of the mini-batches
    :param seed: Seed to set the random state and get reproducibility
    :return: Dictionary with the embeddings and the time expended during the training
    """
    start_time = time()
    w2v_model = Word2Vec(min_count=0, window=win_size, vector_size=emb_size,
                         alpha=learning_rate, min_alpha=min_lr,
                         negative=ns_rate, batch_words=batch_size, sg=0)
    w2v_model.build_vocab(cases)
    w2v_model.train(cases, total_examples=w2v_model.corpus_count, epochs=epochs, report_delay=1)
    end_time = time()

    dict_embeddings = {}
    for i in range(len(w2v_model.wv)):
        dict_embeddings.update({
            i: w2v_model.wv[i]
        })

    return dict_embeddings, end_time - start_time
