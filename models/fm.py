"""Факторизационная машина с метапризнаками пользователей и заведений, функция потерь BPR."""

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from .common import RELEVANT_MIN_STARS, BaseRecommender, IdIndex
from .time_svdpp import to_days

DEFAULT_ATTRIBUTES = (
    "RestaurantsPriceRange2", "NoiseLevel", "Alcohol", "WiFi", "RestaurantsAttire", "OutdoorSeating",
)
USER_NUMERIC = ("account_age_years", "elite_years_before", "user_log_reviews", "user_mean_stars")
ITEM_NUMERIC = ("item_log_reviews", "item_mean_stars")


def clean_attribute(column: str) -> pl.Expr:
    """u'free' / 'free' / 'None' -> free / null."""
    return (
        pl.col(column).cast(pl.String)
        .str.replace_all(r"^u?'|'$", "")
        .replace("None", None)
        .alias(column)
    )


def parse_elite_years(elite, pair_is_2020: bool) -> list:
    """Годы статуса Elite из строки вида '2017,2018,20,20'; пара '20,20' трактуется как 2020 при pair_is_2020."""
    if not elite:
        return []
    tokens = [token.strip() for token in elite.split(",")]
    years = {int(token) for token in tokens if len(token) == 4 and token.isdigit()}
    if pair_is_2020 and tokens.count("20") >= 2:
        years.add(2020)
    return sorted(years)


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.where(std > 0, std, 1.0)


class YelpFeatureBuilder:
    """Разреженные векторы признаков FM для пользователей и заведений Yelp.

    Блоки вектора x: one-hot пользователя | one-hot заведения | категории заведения (несколько единиц) |
    город | значения атрибутов | числовые признаки пользователя | числовые признаки заведения.

    Числовые признаки пользователя в момент t: возраст аккаунта в годах (t − yelping_since), число лет Elite
    строго до года t, log(1 + число отзывов в истории), средняя оценка в истории минус общая средняя.
    Числовые признаки заведения: log(1 + число отзывов в истории), средняя оценка в истории минус общая средняя.
    Агрегаты считаются только по переданной истории (обучающему периоду этапа). Числовые признаки
    стандартизируются: признаки пользователя — по обучающим примерам, признаки заведения — по заведениям.
    """

    def __init__(self, business_data: pl.DataFrame, user_data: pl.DataFrame,
                 attributes=DEFAULT_ATTRIBUTES, elite_pair_is_2020: bool = True, min_token_count: int = 1):
        self.business_data = business_data
        self.user_data = user_data
        self.attributes = tuple(attributes)
        self.elite_pair_is_2020 = elite_pair_is_2020
        self.min_token_count = min_token_count

    def fit(self, history: pl.DataFrame, user_index: IdIndex, item_index: IdIndex):
        self.n_users, self.n_items = len(user_index), len(item_index)
        self.mu = float(history["stars"].mean())

        u = user_index.encode(history["user_id"].to_numpy())
        i = item_index.encode(history["business_id"].to_numpy())
        stars = history["stars"].to_numpy().astype(np.float64)

        item_tokens = self._item_tokens(item_index)
        token_counts = {}
        for tokens in item_tokens:
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        vocab = sorted(t for t, c in token_counts.items() if c >= self.min_token_count)
        base = self.n_users + self.n_items
        self.token_index = {token: base + j for j, token in enumerate(vocab)}
        self.user_numeric_index = base + len(vocab) + np.arange(len(USER_NUMERIC))
        self.item_numeric_index = base + len(vocab) + len(USER_NUMERIC) + np.arange(len(ITEM_NUMERIC))
        self.n_features = base + len(vocab) + len(USER_NUMERIC) + len(ITEM_NUMERIC)
        self.pad_index = self.n_features

        item_count = np.bincount(i, minlength=self.n_items).astype(np.float64)
        item_mean = np.bincount(i, weights=stars, minlength=self.n_items) / np.maximum(item_count, 1)
        item_numeric = np.column_stack([np.log1p(item_count), np.where(item_count > 0, item_mean - self.mu, 0.0)])
        item_numeric = _standardize(item_numeric, item_numeric.mean(0), item_numeric.std(0))

        rows_idx, rows_val = [], []
        for item, tokens in enumerate(item_tokens):
            idx = [self.n_users + item] + [self.token_index[t] for t in tokens if t in self.token_index]
            idx += self.item_numeric_index.tolist()
            val = [1.0] * (len(idx) - len(ITEM_NUMERIC)) + item_numeric[item].tolist()
            rows_idx.append(idx)
            rows_val.append(val)
        width = max(len(r) for r in rows_idx)
        self.item_idx = np.full((self.n_items, width), self.pad_index, dtype=np.int64)
        self.item_val = np.zeros((self.n_items, width), dtype=np.float32)
        for item, (idx, val) in enumerate(zip(rows_idx, rows_val)):
            self.item_idx[item, :len(idx)] = idx
            self.item_val[item, :len(val)] = val

        user_count = np.bincount(u, minlength=self.n_users).astype(np.float64)
        user_mean = np.bincount(u, weights=stars, minlength=self.n_users) / np.maximum(user_count, 1)
        self.user_aggregates = np.column_stack([
            np.log1p(user_count), np.where(user_count > 0, user_mean - self.mu, 0.0),
        ])
        self._fit_user_meta(user_index)
        self.user_mean_ = np.zeros(len(USER_NUMERIC))
        self.user_std_ = np.ones(len(USER_NUMERIC))
        return self

    def _item_tokens(self, item_index: IdIndex) -> list:
        frame = pl.DataFrame({"business_id": item_index.ids, "item_idx": np.arange(len(item_index))})
        columns = ["business_id"] + [c for c in ("categories", "city", "attributes") if c in self.business_data.columns]
        meta = frame.join(self.business_data.select(columns), on="business_id", how="left").sort("item_idx")

        tokens = [[] for _ in range(len(item_index))]
        if "categories" in meta.columns:
            categories = meta.select(
                pl.col("categories").str.split(",").list.eval(pl.element().str.strip_chars())
            )["categories"].to_list()
            for item, values in enumerate(categories):
                tokens[item] += [f"category={v}" for v in (values or []) if v]
        if "city" in meta.columns:
            for item, value in enumerate(meta["city"].to_list()):
                if value:
                    tokens[item].append(f"city={value}")
        if "attributes" in meta.columns:
            fields = {f.name for f in meta.schema["attributes"].fields}
            attrs = meta.select(pl.col("attributes").struct.unnest())
            for name in self.attributes:
                if name not in fields:
                    continue
                for item, value in enumerate(attrs.select(clean_attribute(name))[name].to_list()):
                    if value is not None:
                        tokens[item].append(f"{name}={value}")
        return tokens

    def _fit_user_meta(self, user_index: IdIndex) -> None:
        frame = pl.DataFrame({"user_id": user_index.ids, "user_idx": np.arange(len(user_index))})
        meta = frame.join(self.user_data.select("user_id", "yelping_since", "elite"), on="user_id", how="left")
        meta = meta.sort("user_idx")
        since = meta["yelping_since"]
        if since.dtype == pl.String:
            since = meta.select(pl.coalesce(
                pl.col("yelping_since").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
                pl.col("yelping_since").str.to_datetime("%Y-%m-%d", strict=False),
            ))["yelping_since"]
        self.signup_days = to_days(since)

        elite = [parse_elite_years(e, self.elite_pair_is_2020) for e in meta["elite"].to_list()]
        width = max([len(e) for e in elite] + [1])
        self.elite_years = np.full((len(elite), width), 10_000, dtype=np.int32)
        for row, years in enumerate(elite):
            self.elite_years[row, :len(years)] = years

    def user_numeric_raw(self, user_idx: np.ndarray, t_days: np.ndarray) -> np.ndarray:
        """Нестандартизованные числовые признаки пользователей в моменты t_days."""
        age = np.nan_to_num((t_days - self.signup_days[user_idx]) / 365.25, nan=0.0).clip(min=0.0)
        year = (np.datetime64("1970-01-01") + t_days.astype("timedelta64[D]")).astype("datetime64[Y]")
        year = year.astype(np.int64) + 1970
        elite_before = (self.elite_years[user_idx] < year[:, None]).sum(1)
        return np.column_stack([age, elite_before, self.user_aggregates[user_idx]])

    def fit_user_scaler(self, raw: np.ndarray) -> None:
        self.user_mean_, self.user_std_ = raw.mean(0), raw.std(0)

    def user_arrays(self, user_idx: np.ndarray, t_days: np.ndarray):
        """Индексы и значения ненулевых признаков пользователей: one-hot и числовые признаки."""
        raw = self.user_numeric_raw(user_idx, t_days)
        values = _standardize(raw, self.user_mean_, self.user_std_)
        idx = np.column_stack([user_idx, np.tile(self.user_numeric_index, (len(user_idx), 1))])
        val = np.column_stack([np.ones(len(user_idx)), values]).astype(np.float32)
        return idx.astype(np.int64), val


class _FMNet(nn.Module):
    def __init__(self, n_features: int, n_factors: int):
        super().__init__()
        self.w = nn.Embedding(n_features + 1, 1, padding_idx=n_features)
        self.V = nn.Embedding(n_features + 1, n_factors, padding_idx=n_features)
        nn.init.zeros_(self.w.weight)
        nn.init.normal_(self.V.weight, std=0.01)
        with torch.no_grad():
            self.V.weight[n_features].zero_()

    def parts(self, idx: torch.Tensor, val: torch.Tensor):
        """Сумма векторов признаков e = Σ v_k x_k и собственный вклад блока: Σ w_k x_k + попарные взаимодействия внутри блока."""
        v = self.V(idx) * val.unsqueeze(-1)
        e = v.sum(1)
        linear = (self.w(idx).squeeze(-1) * val).sum(1)
        within = 0.5 * (e.pow(2).sum(1) - v.pow(2).sum((1, 2)))
        return e, linear + within


class FactorizationMachineBPR(BaseRecommender):
    """Факторизационная машина второго порядка с функцией потерь BPR.

    Модель: ŷ(x) = w_0 + Σ_k w_k x_k + Σ_{k<l} ⟨v_k, v_l⟩ x_k x_l, где x — разреженный вектор признаков пары
    (пользователь в момент t, заведение), построенный YelpFeatureBuilder. Вектор делится на блок пользователя
    и блок заведения; ŷ = const(u) + s_i + ⟨e_u, e_i⟩, где e — сумма v_k x_k по блоку, s_i — линейные члены
    и взаимодействия внутри блока заведения. Слагаемые, зависящие только от пользователя, не влияют
    на ранжирование заведений и сокращаются в BPR.

    Обучение (BPR): для каждого положительного отзыва (оценка ≥ RELEVANT_MIN_STARS) пользователя u в момент t
    выбирается отрицательное заведение j равномерно из всех заведений истории; минимизируется
        −log σ(ŷ(u, t, i) − ŷ(u, t, j)) + reg · (||v||² + ||w||² для признаков примера) оптимизатором Adam.
    Рекомендация: признаки пользователя вычисляются в момент as_of.

    Параметры: business_data, user_data — таблицы метаданных Yelp; n_factors, n_epochs, lr, reg, batch_size;
    attributes — атрибуты заведений; elite_pair_is_2020 — трактовка пары '20,20' в поле elite;
    device (по умолчанию cuda при наличии); random_state.
    """

    def __init__(self, business_data: pl.DataFrame, user_data: pl.DataFrame, n_factors: int = 32,
                 n_epochs: int = 20, lr: float = 1e-3, reg: float = 1e-5, batch_size: int = 4096,
                 attributes=DEFAULT_ATTRIBUTES, elite_pair_is_2020: bool = True,
                 device: str | None = None, random_state: int = 42):
        self.business_data = business_data
        self.user_data = user_data
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.attributes = attributes
        self.elite_pair_is_2020 = elite_pair_is_2020
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    def fit(self, history: pl.DataFrame, as_of=None):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self.user_index = IdIndex(history["user_id"].to_numpy())
        self.item_index = IdIndex(history["business_id"].to_numpy())
        self.features = YelpFeatureBuilder(
            self.business_data, self.user_data, self.attributes, self.elite_pair_is_2020,
        ).fit(history, self.user_index, self.item_index)
        self.as_of_days = float(to_days([as_of])[0]) if as_of is not None else float(to_days(history["date"]).max())

        positives = history.filter(pl.col("stars") >= RELEVANT_MIN_STARS)
        if positives.height == 0:
            raise ValueError("В истории нет положительных отзывов для обучения BPR")
        pos_u = self.user_index.encode(positives["user_id"].to_numpy())
        pos_i = self.item_index.encode(positives["business_id"].to_numpy())
        pos_t = to_days(positives["date"])
        self.features.fit_user_scaler(self.features.user_numeric_raw(pos_u, pos_t))
        user_idx, user_val = self.features.user_arrays(pos_u, pos_t)

        dev_ = self.device
        self.net = _FMNet(self.features.n_features, self.n_factors).to(dev_)
        U_idx = torch.tensor(user_idx, device=dev_)
        U_val = torch.tensor(user_val, device=dev_)
        I_idx = torch.tensor(self.features.item_idx, device=dev_)
        I_val = torch.tensor(self.features.item_val, device=dev_)
        P_i = torch.tensor(pos_i, dtype=torch.long, device=dev_)
        n_items = len(self.item_index)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_loss_ = []
        n = len(pos_u)
        for _ in range(self.n_epochs):
            perm = torch.tensor(rng.permutation(n), dtype=torch.long, device=dev_)
            neg_all = torch.tensor(rng.integers(0, n_items, size=n), dtype=torch.long, device=dev_)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                ui, uv = U_idx[b], U_val[b]
                pi, ni = P_i[b], neg_all[b]
                e_u, _ = self.net.parts(ui, uv)
                e_p, s_p = self.net.parts(I_idx[pi], I_val[pi])
                e_n, s_n = self.net.parts(I_idx[ni], I_val[ni])
                diff = s_p - s_n + (e_u * (e_p - e_n)).sum(1)
                used = torch.cat([ui, I_idx[pi], I_idx[ni]], dim=1)
                reg = self.net.V(used).pow(2).sum((1, 2)) + self.net.w(used).pow(2).sum((1, 2))
                loss = -F.logsigmoid(diff).mean() + self.reg * reg.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(b)
            self.train_loss_.append(total / n)

        with torch.no_grad():
            self._item_e, self._item_s = self.net.parts(I_idx, I_val)
        return self

    @torch.no_grad()
    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        user_idx = np.asarray(user_idx)
        t = np.full(len(user_idx), self.as_of_days)
        idx, val = self.features.user_arrays(user_idx, t)
        e_u, _ = self.net.parts(torch.tensor(idx, device=self.device), torch.tensor(val, device=self.device))
        scores = self._item_s.unsqueeze(0) + e_u @ self._item_e.T
        return scores.cpu().numpy()
