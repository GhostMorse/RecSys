"""Коллаборативная фильтрация на основе соседей: item-based kNN с косинусной мерой сходства."""

import numpy as np
import polars as pl
from scipy import sparse

from .common import BaseRecommender, IdIndex, interaction_matrix


class ItemKNN(BaseRecommender):
    """Item-based kNN.

    Сходство заведений i и j — косинус между столбцами матрицы взаимодействий X:
        sim(i, j) = x_i · x_j / (||x_i|| · ||x_j||).
    Для каждого заведения i сохраняются k_neighbors наиболее похожих заведений N_k(i).
    Оценка заведения i для пользователя u:
        score(u, i) = Σ_{j ∈ N_k(i) ∩ I_u} sim(i, j) · x_uj,
    где I_u — заведения из истории пользователя, x_uj — значение сигнала.

    Параметры: k_neighbors — размер окрестности; signal — сигнал матрицы взаимодействий
    (all_binary, positive_binary, rating); block_size — число заведений в блоке при вычислении сходств.
    """

    def __init__(self, k_neighbors: int = 50, signal: str = "all_binary", block_size: int = 1024):
        self.k_neighbors = k_neighbors
        self.signal = signal
        self.block_size = block_size

    def fit(self, history: pl.DataFrame, as_of=None):
        self.user_index = IdIndex(history["user_id"].to_numpy())
        self.item_index = IdIndex(history["business_id"].to_numpy())
        self.X = interaction_matrix(history, self.user_index, self.item_index, self.signal)

        n_items = len(self.item_index)
        k = min(self.k_neighbors, n_items - 1)
        norms = np.sqrt(np.asarray(self.X.multiply(self.X).sum(axis=0)).ravel())
        inv_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        X_norm = (self.X @ sparse.diags(inv_norms.astype(np.float32))).tocsc()
        X_norm_t = X_norm.T.tocsr()

        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block_size):
            end = min(start + self.block_size, n_items)
            sim = (X_norm_t @ X_norm[:, start:end]).toarray()
            sim[np.arange(start, end), np.arange(end - start)] = 0.0
            if k <= 0:
                continue
            neighbors = np.argpartition(-sim, k - 1, axis=0)[:k, :]
            values = np.take_along_axis(sim, neighbors, axis=0)
            rows.append(neighbors.ravel(order="F"))
            cols.append(np.repeat(np.arange(start, end), k))
            vals.append(values.ravel(order="F"))

        if rows:
            similarity = sparse.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n_items, n_items),
                dtype=np.float32,
            )
            similarity.eliminate_zeros()
        else:
            similarity = sparse.csr_matrix((n_items, n_items), dtype=np.float32)
        self.similarity = similarity
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return (self.X[user_idx] @ self.similarity).toarray()
