"""EASE — Embarrassingly Shallow Autoencoder (Steck, 2019)."""

import numpy as np
import polars as pl

from .common import BaseRecommender, IdIndex, interaction_matrix


class EASE(BaseRecommender):
    """EASE.

    Задача: min_B ||X − XB||²_F + λ||B||²_F при условии diag(B) = 0.
    Решение в замкнутой форме: P = (XᵀX + λI)⁻¹, B_ij = −P_ij / P_jj при i ≠ j, B_jj = 0.
    Оценки пользователя: x_u B.

    Параметры: reg — коэффициент регуляризации λ; signal — сигнал матрицы взаимодействий
    (all_binary, positive_binary, rating).
    """

    def __init__(self, reg: float = 500.0, signal: str = "all_binary"):
        self.reg = reg
        self.signal = signal

    def fit(self, history: pl.DataFrame, as_of=None):
        self.user_index = IdIndex(history["user_id"].to_numpy())
        self.item_index = IdIndex(history["business_id"].to_numpy())
        X = interaction_matrix(history, self.user_index, self.item_index, self.signal, dtype=np.float64)

        gram = (X.T @ X).toarray()
        gram[np.diag_indices_from(gram)] += self.reg
        P = np.linalg.inv(gram)
        del gram
        P /= -np.diag(P)[np.newaxis, :]
        np.fill_diagonal(P, 0.0)

        self.B = P.astype(np.float32)
        del P
        self.X = X.astype(np.float32).tocsr()
        return self

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[user_idx] @ self.B)
