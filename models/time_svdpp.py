"""timeSVD++ — матричная факторизация с временной динамикой (Koren, 2009)."""

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from .common import BaseRecommender, IdIndex


def to_days(values) -> np.ndarray:
    """Даты (Series, список или скаляр datetime) в дни от 1970-01-01."""
    series = values if isinstance(values, pl.Series) else pl.Series(list(values))
    return series.dt.epoch("s").to_numpy().astype(np.float64) / 86400.0


class _TimeSVDppNet(nn.Module):
    def __init__(self, n_users, n_items, n_user_days, n_bins, n_factors, mu, indptr, indices):
        super().__init__()
        self.n_items = n_items
        self.n_bins = n_bins
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("indptr", indptr)
        self.register_buffer("indices", indices)

        self.b_u = nn.Embedding(n_users, 1)
        self.alpha_u = nn.Embedding(n_users, 1)
        self.b_ut = nn.Embedding(n_user_days, 1)
        self.c_u = nn.Embedding(n_users, 1)
        self.c_ut = nn.Embedding(n_user_days, 1)
        self.b_i = nn.Embedding(n_items, 1)
        self.b_ibin = nn.Embedding(n_items * n_bins, 1)
        self.P = nn.Embedding(n_users, n_factors)
        self.alpha_p = nn.Embedding(n_users, n_factors)
        self.Q = nn.Embedding(n_items, n_factors)
        self.Y = nn.Embedding(n_items, n_factors)

        for emb in (self.b_u, self.alpha_u, self.b_ut, self.c_ut, self.b_i, self.b_ibin, self.alpha_p):
            nn.init.zeros_(emb.weight)
        nn.init.ones_(self.c_u.weight)
        for emb in (self.P, self.Q, self.Y):
            nn.init.normal_(emb.weight, std=0.01)

    def implicit(self, users: torch.Tensor):
        """|N(u)|^(-1/2) Σ_{j ∈ N(u)} y_j и (1/|N(u)|) Σ_{j ∈ N(u)} ||y_j||² для каждого пользователя."""
        starts = self.indptr[users]
        lengths = self.indptr[users + 1] - starts
        offsets = torch.cumsum(lengths, 0) - lengths
        total = int(lengths.sum().item())
        positions = torch.arange(total, device=users.device) - torch.repeat_interleave(offsets, lengths)
        flat = self.indices[torch.repeat_interleave(starts, lengths) + positions]
        sums = F.embedding_bag(flat, self.Y.weight, offsets, mode="sum")
        sq_norms = F.embedding_bag(flat, self.Y.weight.pow(2).sum(1, keepdim=True), offsets, mode="sum")
        lengths_f = lengths.clamp(min=1).to(sums.dtype).unsqueeze(1)
        return sums / lengths_f.sqrt(), sq_norms / lengths_f

    def forward(self, u, i, dev, bins, ud):
        unique_users, inverse = torch.unique(u, return_inverse=True)
        imp, imp_sq = self.implicit(unique_users)
        imp, imp_sq = imp[inverse], imp_sq[inverse]

        dev = dev.unsqueeze(1)
        b_u, alpha_u, b_ut = self.b_u(u), self.alpha_u(u), self.b_ut(ud)
        c_u, c_ut = self.c_u(u), self.c_ut(ud)
        b_i, b_ibin = self.b_i(i), self.b_ibin(i * self.n_bins + bins)
        p_u, alpha_p, q_i = self.P(u), self.alpha_p(u), self.Q(i)

        bias = self.mu + b_u + alpha_u * dev + b_ut + (b_i + b_ibin) * (c_u + c_ut)
        user_vec = p_u + alpha_p * dev + imp
        pred = bias.squeeze(1) + (q_i * user_vec).sum(1)

        reg_terms = {
            "bias": (b_u.pow(2) + b_i.pow(2) + b_ibin.pow(2) + (c_u - 1).pow(2)).squeeze(1),
            "day": (b_ut.pow(2) + c_ut.pow(2)).squeeze(1),
            "time": alpha_u.pow(2).squeeze(1) + alpha_p.pow(2).sum(1),
            "factors": p_u.pow(2).sum(1) + q_i.pow(2).sum(1) + imp_sq.squeeze(1),
        }
        return pred, reg_terms


class TimeSVDpp(BaseRecommender):
    """timeSVD++.

    Предсказание оценки пользователя u заведению i в момент t:
        r̂_ui(t) = μ + b_u + α_u·dev_u(t) + b_{u,t} + (b_i + b_{i,Bin(t)})·(c_u + c_{u,t})
                  + q_iᵀ (p_u + α^p_u·dev_u(t) + |N(u)|^(-1/2) Σ_{j ∈ N(u)} y_j),
    где dev_u(t) = sign(t − t_u)·|t − t_u|^β, t_u — средняя дата оценок пользователя (в днях),
    Bin(t) — номер временного интервала (n_bins интервалов равной длины на обучающем периоде),
    b_{u,t} и c_{u,t} — параметры дня (пользователь, календарный день), N(u) — заведения из истории u.
    Обучение: минимизация среднего квадрата ошибки на оценках истории с L2-регуляризацией
    (reg_bias — смещения b_u, b_i, b_{i,Bin}, c_u; reg_day — дневные параметры b_{u,t}, c_{u,t};
    reg_time — коэффициенты α_u, α^p_u; reg_factors — векторы p, q, y) оптимизатором Adam.
    Рекомендация в момент as_of: дневные параметры равны нулю (день не встречался при обучении),
    Bin(as_of) ограничивается последним интервалом.

    Параметры: n_factors, n_epochs, lr, batch_size, reg_bias, reg_day, reg_time, reg_factors, n_bins, beta,
    device (по умолчанию cuda при наличии), random_state.
    """

    def __init__(self, n_factors: int = 32, n_epochs: int = 5, lr: float = 5e-3, batch_size: int = 8192,
                 reg_bias: float = 0.05, reg_day: float = 0.05, reg_time: float = 10.0, reg_factors: float = 0.05,
                 n_bins: int = 30,
                 beta: float = 0.4, device: str | None = None, random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.reg_bias = reg_bias
        self.reg_day = reg_day
        self.reg_time = reg_time
        self.reg_factors = reg_factors
        self.n_bins = n_bins
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    def _dev(self, t_days: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
        delta = t_days - self.user_mean_day[users]
        return torch.sign(delta) * delta.abs().pow(self.beta)

    def _bin(self, t_days: torch.Tensor) -> torch.Tensor:
        width = (self.t_max - self.t_min) / self.n_bins + 1e-9
        return ((t_days - self.t_min) / width).floor().clamp(0, self.n_bins - 1).long()

    def fit(self, history: pl.DataFrame, as_of=None):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self.user_index = IdIndex(history["user_id"].to_numpy())
        self.item_index = IdIndex(history["business_id"].to_numpy())
        u = self.user_index.encode(history["user_id"].to_numpy())
        i = self.item_index.encode(history["business_id"].to_numpy())
        r = history["stars"].to_numpy().astype(np.float32)
        t = to_days(history["date"])
        n_users, n_items = len(self.user_index), len(self.item_index)

        day = np.floor(t).astype(np.int64)
        user_day_keys, ud = np.unique(u * (day.max() + 1) + day, return_inverse=True)

        order = np.argsort(u, kind="stable")
        counts = np.bincount(u, minlength=n_users)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        indices = i[order]

        user_mean_day = np.bincount(u, weights=t, minlength=n_users) / np.maximum(counts, 1)
        self.t_min, self.t_max = float(t.min()), float(t.max())
        self.as_of_days = float(to_days([as_of])[0]) if as_of is not None else self.t_max

        dev_ = self.device
        self.user_mean_day = torch.tensor(user_mean_day, dtype=torch.float32, device=dev_)
        self.net = _TimeSVDppNet(
            n_users, n_items, len(user_day_keys), self.n_bins, self.n_factors, float(r.mean()),
            torch.tensor(indptr, dtype=torch.long), torch.tensor(indices, dtype=torch.long),
        ).to(dev_)

        u_t = torch.tensor(u, dtype=torch.long, device=dev_)
        i_t = torch.tensor(i, dtype=torch.long, device=dev_)
        r_t = torch.tensor(r, dtype=torch.float32, device=dev_)
        t_t = torch.tensor(t, dtype=torch.float32, device=dev_)
        ud_t = torch.tensor(ud, dtype=torch.long, device=dev_)
        dev_all = self._dev(t_t, u_t)
        bins_all = self._bin(t_t)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.train_loss_ = []
        n = len(r)
        for _ in range(self.n_epochs):
            perm = torch.tensor(rng.permutation(n), dtype=torch.long, device=dev_)
            total, mse_total = 0.0, 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                pred, reg = self.net(u_t[b], i_t[b], dev_all[b], bins_all[b], ud_t[b])
                mse = (pred - r_t[b]).pow(2).mean()
                loss = mse + (self.reg_bias * reg["bias"] + self.reg_day * reg["day"] + self.reg_time * reg["time"]
                              + self.reg_factors * reg["factors"]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(b)
                mse_total += mse.item() * len(b)
            self.train_loss_.append({"loss": total / n, "rmse": float(np.sqrt(mse_total / n))})
        return self

    @torch.no_grad()
    def predict_scores(self, user_idx: np.ndarray, at_days: float | None = None) -> np.ndarray:
        """Предсказанные оценки всех заведений для пользователей в момент at_days (по умолчанию as_of)."""
        net = self.net
        at_days = self.as_of_days if at_days is None else at_days
        users = torch.tensor(np.asarray(user_idx), dtype=torch.long, device=self.device)
        t = torch.full((len(users),), at_days, dtype=torch.float32, device=self.device)
        dev = self._dev(t, users).unsqueeze(1)
        bin_t = int(self._bin(torch.tensor([at_days], device=self.device)).item())

        imp, _ = net.implicit(users)
        user_vec = net.P(users) + net.alpha_p(users) * dev + imp
        item_bias = net.b_i.weight[:, 0] + net.b_ibin.weight.view(net.n_items, net.n_bins)[:, bin_t]
        scores = (net.mu + net.b_u(users) + net.alpha_u(users) * dev
                  + item_bias.unsqueeze(0) * net.c_u(users)
                  + user_vec @ net.Q.weight.T)
        return scores.cpu().numpy()

    def _score(self, user_idx: np.ndarray) -> np.ndarray:
        return self.predict_scores(user_idx)
