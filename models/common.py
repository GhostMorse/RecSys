"""Общие компоненты моделей: индексация идентификаторов, матрица взаимодействий, формирование топ-K."""

import numpy as np
import polars as pl
from scipy import sparse

SIGNALS = ("all_binary", "positive_binary", "rating")
RELEVANT_MIN_STARS = 4


class IdIndex:
    """Взаимно однозначное соответствие исходных идентификаторов и индексов 0..n-1.

    Идентификаторы хранятся в отсортированном виде, поэтому индексация детерминирована.
    """

    def __init__(self, ids):
        self.ids = np.unique(np.asarray(ids, dtype=str))

    def __len__(self):
        return int(self.ids.size)

    def encode(self, values) -> np.ndarray:
        """Индексы значений; для неизвестных значений возвращается -1."""
        values = np.asarray(values, dtype=str)
        if values.size == 0 or self.ids.size == 0:
            return np.full(values.shape, -1, dtype=np.int64)
        pos = np.clip(np.searchsorted(self.ids, values), 0, self.ids.size - 1)
        return np.where(self.ids[pos] == values, pos, -1).astype(np.int64)

    def decode(self, idx) -> np.ndarray:
        return self.ids[np.asarray(idx, dtype=np.int64)]


def signal_frame(df: pl.DataFrame, signal: str) -> pl.DataFrame:
    """Взаимодействия и значения сигнала.

    all_binary — все отзывы со значением 1; positive_binary — только отзывы с оценкой не ниже
    RELEVANT_MIN_STARS со значением 1; rating — все отзывы со значением, равным оценке.
    """
    if signal not in SIGNALS:
        raise ValueError(f"signal должен быть одним из {SIGNALS}, получено: {signal}")
    if signal == "positive_binary":
        df = df.filter(pl.col("stars") >= RELEVANT_MIN_STARS)
    value = pl.col("stars").cast(pl.Float64) if signal == "rating" else pl.lit(1.0)
    return df.select("user_id", "business_id", value.alias("value"))


def interaction_matrix(df: pl.DataFrame, user_index: IdIndex, item_index: IdIndex, signal: str,
                       dtype=np.float32) -> sparse.csr_matrix:
    """Разреженная матрица «пользователь × заведение» со значениями выбранного сигнала."""
    frame = signal_frame(df, signal)
    rows = user_index.encode(frame["user_id"].to_numpy())
    cols = item_index.encode(frame["business_id"].to_numpy())
    data = frame["value"].to_numpy().astype(dtype)
    known = (rows >= 0) & (cols >= 0)
    return sparse.csr_matrix(
        (data[known], (rows[known], cols[known])),
        shape=(len(user_index), len(item_index)),
        dtype=dtype,
    )


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k наибольших значений в каждой строке, упорядоченные по убыванию."""
    n_items = scores.shape[1]
    k = min(k, n_items)
    if k == n_items:
        part = np.tile(np.arange(n_items), (scores.shape[0], 1))
    else:
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


class BaseRecommender:
    """Базовый класс: единые правила формирования рекомендаций для всех моделей.

    Наследник реализует fit и _score(user_idx) -> np.ndarray формы (len(user_idx), число заведений).
    Рекомендации строятся только из набора кандидатов, уже известные пользователю заведения исключаются.
    """

    user_index: IdIndex
    item_index: IdIndex

    def fit(self, history: pl.DataFrame, as_of=None):
        raise NotImplementedError

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_ids, candidates, seen: dict, k: int = 10, batch_size: int = 2048) -> dict:
        """Топ-k заведений для каждого пользователя.

        user_ids — идентификаторы пользователей; candidates — идентификаторы заведений-кандидатов;
        seen — словарь «пользователь → множество известных заведений». Пользователям, неизвестным
        модели, соответствует пустой список.
        """
        user_ids = list(user_ids)
        candidate_mask = np.zeros(len(self.item_index), dtype=bool)
        candidate_idx = self.item_index.encode(list(candidates))
        candidate_mask[candidate_idx[candidate_idx >= 0]] = True

        user_idx = self.user_index.encode(user_ids)
        result = {u: [] for u, idx in zip(user_ids, user_idx) if idx < 0}
        known_users = [(u, idx) for u, idx in zip(user_ids, user_idx) if idx >= 0]

        for start in range(0, len(known_users), batch_size):
            batch = known_users[start:start + batch_size]
            scores = np.asarray(self._score(np.array([idx for _, idx in batch])), dtype=np.float64)
            scores[:, ~candidate_mask] = -np.inf
            for row, (user, _) in enumerate(batch):
                known_items = seen.get(user)
                if known_items:
                    seen_idx = self.item_index.encode(list(known_items))
                    scores[row, seen_idx[seen_idx >= 0]] = -np.inf
            best = top_k(scores, k)
            for row, (user, _) in enumerate(batch):
                items = best[row][np.isfinite(scores[row, best[row]])]
                result[user] = self.item_index.decode(items).tolist()
        return result
