"""iALS — ALS для неявной обратной связи (Hu, Koren, Volinsky, 2008)."""

import numpy as np
import polars as pl
from scipy import linalg

from .common import BaseRecommender, IdIndex, interaction_matrix


class ImplicitALS(BaseRecommender):
    """iALS.

    Предпочтение p_ui = 1, если значение сигнала r_ui > 0, иначе 0; уверенность c_ui = 1 + α r_ui.
    Задача: min_{U,V} Σ_{u,i} c_ui (p_ui − u_uᵀ v_i)² + λ (||U||²_F + ||V||²_F).
    Шаг по пользователю u при фиксированной V:
        u_u = (VᵀV + V_{I_u}ᵀ diag(α r_u) V_{I_u} + λI)⁻¹ V_{I_u}ᵀ (1 + α r_u),
    шаг по заведению — симметрично. Оценки пользователя: V u_u.

    Параметры: n_factors — размерность векторов; n_iters — число итераций ALS; reg — λ; alpha — α;
    signal — сигнал r_ui (all_binary, positive_binary, rating); random_state — зерно инициализации.
    """

    def __init__(self, n_factors: int = 64, n_iters: int = 15, reg: float = 0.1, alpha: float = 10.0,
                 signal: str = "all_binary", random_state: int = 42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.alpha = alpha
        self.signal = signal
        self.random_state = random_state

    def fit(self, history: pl.DataFrame, as_of=None):
        self.user_index = IdIndex(history["user_id"].to_numpy())
        self.item_index = IdIndex(history["business_id"].to_numpy())
        R = interaction_matrix(history, self.user_index, self.item_index, self.signal, dtype=np.float64)
        R_t = R.T.tocsr()

        rng = np.random.default_rng(self.random_state)
        self.user_factors = 0.01 * rng.standard_normal((R.shape[0], self.n_factors))
        self.item_factors = 0.01 * rng.standard_normal((R.shape[1], self.n_factors))

        for _ in range(self.n_iters):
            self._solve(R, self.item_factors, self.user_factors)
            self._solve(R_t, self.user_factors, self.item_factors)
        return self

    def _solve(self, matrix, fixed: np.ndarray, out: np.ndarray) -> None:
        gram = fixed.T @ fixed
        reg_eye = self.reg * np.eye(self.n_factors)
        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
        for row in range(matrix.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                out[row] = 0.0
                continue
            factors = fixed[indices[start:end]]
            confidence = self.alpha * data[start:end]
            A = gram + (factors.T * confidence) @ factors + reg_eye
            b = factors.T @ (1.0 + confidence)
            out[row] = linalg.solve(A, b, assume_a="pos", check_finite=False)

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.user_factors[user_idx] @ self.item_factors.T
