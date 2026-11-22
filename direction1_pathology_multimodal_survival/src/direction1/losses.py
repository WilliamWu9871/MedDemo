"""Discrete-time survival binning and NLLSurvLoss matching mahmoodlab/SurvPath."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_survival_bins(
    times: np.ndarray,
    events: np.ndarray,
    n_bins: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """Fit quantile bin edges strictly on uncensored (events == 1) training survival times.

    Follows SurvPath/MCAT discretization so each discrete hazard bin contains an
    approximately balanced number of observed events without validation leakage.
    Returns an array of length ``n_bins + 1`` covering ``[-inf, ..., +inf]``.
    """
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    uncensored_times = times[events == 1]
    ref_times = uncensored_times if len(uncensored_times) >= n_bins else times
    if len(ref_times) == 0:
        raise ValueError("Cannot fit survival bins on empty time array.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    interior = np.quantile(ref_times, quantiles[1:-1])
    # Ensure strictly monotonic interior edges even when duplicate event times exist
    for i in range(1, len(interior)):
        if interior[i] <= interior[i - 1]:
            interior[i] = interior[i - 1] + eps

    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    return edges


def transform_survival_bins(times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous survival times to discrete bin indices in ``[0, n_bins - 1]``."""
    times = np.asarray(times, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = len(bin_edges) - 1
    interior = bin_edges[1:-1]
    y_disc = np.digitize(times, interior, right=False)
    return np.clip(y_disc, 0, n_bins - 1).astype(np.int64)


def nll_loss(
    h: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    alpha: float = 0.0,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """Discrete-time negative log-likelihood survival loss (Zadeh & Schmid, 2020).

    Matches ``mahmoodlab/SurvPath/utils/loss_func.py``.

    Args:
        h: Logits of shape ``(batch_size, n_bins)`` where ``hazards = sigmoid(h)``.
        y: Discrete time-bin indices of shape ``(batch_size, 1)``.
        c: Censorship indicator of shape ``(batch_size, 1)`` with
           **1 = censored (alive at last follow-up), 0 = uncensored event (death)**.
        alpha: Extra weight placed on uncensored event loss.
        eps: Numerical clamp floor for log safety.
        reduction: ``'mean'`` or ``'sum'``.
    """
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    if not torch.all((c == 0) | (c == 1)):
        raise ValueError("censorship indicator c must contain only 0 (event) and 1 (censored).")

    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    s_padded = torch.cat([torch.ones_like(c, dtype=survival.dtype), survival], dim=1)

    s_prev = torch.gather(s_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(s_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1.0 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss if alpha is not None else neg_l

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction {reduction!r}; choose 'mean' or 'sum'.")


class NLLSurvLoss(nn.Module):
    """Module wrapper around discrete-time NLL survival loss (SurvPath convention).

    Expects ``censorship = 1 - OS_event`` (1 = censored, 0 = death event).
    """

    def __init__(self, alpha: float = 0.15, eps: float = 1e-7, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        h: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(dim=1)
        if c.ndim == 1:
            c = c.unsqueeze(dim=1)
        return nll_loss(
            h=h,
            y=y,
            c=c,
            alpha=self.alpha,
            eps=self.eps,
            reduction=self.reduction,
        )


def hazards_to_survival_and_risk(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert discrete hazard logits ``(B, K)`` to ``(hazards, survival, risk)``.

    ``risk = -sum_k S(k)`` so higher values always correspond to higher mortality risk
    (shorter expected survival), matching C-index ordering against survival time.
    """
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=-1)
    risk = -torch.sum(survival, dim=-1)
    return hazards, survival, risk
