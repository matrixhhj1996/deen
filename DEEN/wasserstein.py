"""2-Wasserstein distance between diagonal Gaussians + Sigma anti-collapse reg.

For two diagonal Gaussians N(mu1, diag(sigma1^2)) and N(mu2, diag(sigma2^2)):

    W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2

Closed-form, differentiable, robust to non-overlapping supports (unlike KL).

Pairing strategy
----------------
In AGW's training loop, each batch contains `num_pos` RGB and `num_pos` IR
samples for each of `batch_size / num_pos` identities, stacked as
`feat = cat(rgb_feats, ir_feats)`. So mu[:N] are RGB, mu[N:] are IR, and
positions i and i+N share the same identity ordering by construction. We
exploit this to pair RGB sample i with IR sample i — same identity pair.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def w2_diagonal_gaussian(mu1: torch.Tensor, log_var1: torch.Tensor,
                         mu2: torch.Tensor, log_var2: torch.Tensor) -> torch.Tensor:
    """Per-dimension mean of W2^2. Inputs (B, D); returns (B,).

    Using mean (not sum) over D makes the loss intrinsically dimension-agnostic
    — λ_w2 doesn't need re-tuning when embed_dim changes, and the loss is
    comparable in scale to per-sample CE / triplet losses.
    """
    sigma1 = torch.exp(0.5 * log_var1)
    sigma2 = torch.exp(0.5 * log_var2)
    mean_term = (mu1 - mu2).pow(2).mean(dim=1)
    var_term = (sigma1 - sigma2).pow(2).mean(dim=1)
    return mean_term + var_term


class CrossModalW2Loss(nn.Module):
    """Aligns RGB and IR distributions of same-identity samples via W2^2.

    The training batch is `[rgb_chunk ; ir_chunk]` with matched ordering, so
    the loss pairs index i (RGB) with index i (IR within the second half).
    """

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        n_total = mu.shape[0]
        assert n_total % 2 == 0, f"expected even batch, got {n_total}"
        n = n_total // 2
        mu_rgb, mu_ir = mu[:n], mu[n:]
        lv_rgb, lv_ir = log_var[:n], log_var[n:]
        w2_sq = w2_diagonal_gaussian(mu_rgb, lv_rgb, mu_ir, lv_ir)
        return w2_sq.mean()


def sym_kl_diagonal_gaussian(mu1: torch.Tensor, log_var1: torch.Tensor,
                             mu2: torch.Tensor, log_var2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two diagonal Gaussians, per-dim mean.

    SymKL = 0.5 (KL(p||q) + KL(q||p))
          = 0.25 sum_i [ var1/var2 + var2/var1
                         + (mu1-mu2)^2 (1/var1 + 1/var2)
                         - 2 ]
    The log terms cancel under symmetrization. The remaining terms diverge
    as variance → 0 while means differ — exactly the σ-collapse pressure that
    plain W2 lacks (Stochastic-WAE / PCME++ analysis).

    Per-dim mean → dimension-agnostic (same convention as W2 above).
    """
    var1 = log_var1.exp()
    var2 = log_var2.exp()
    inv_var1 = (-log_var1).exp()
    inv_var2 = (-log_var2).exp()
    diff_sq = (mu1 - mu2).pow(2)
    sym_kl = 0.25 * (var1 * inv_var2 + var2 * inv_var1
                     + diff_sq * (inv_var1 + inv_var2) - 2.0)
    return sym_kl.mean(dim=1)


class CrossModalSymKLLoss(nn.Module):
    """Same RGB↔IR pairing as CrossModalW2Loss, but with symmetric KL distance.

    By default mu is detached: sym-KL acts as a pure σ regulator. The (μ-μ)²/σ²
    term still flows back to log_var (so σ must grow when means disagree), but
    its gradient does NOT push μ — that path causes runaway explosion because
    inv_var amplifies μ-gradients by ~200x at init. μ alignment is delegated
    to the ID losses (CE on mu_classifier(mu)).
    """

    def __init__(self, detach_mu: bool = True):
        super().__init__()
        self.detach_mu = detach_mu

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        n_total = mu.shape[0]
        assert n_total % 2 == 0, f"expected even batch, got {n_total}"
        n = n_total // 2
        mu_rgb, mu_ir = mu[:n], mu[n:]
        if self.detach_mu:
            mu_rgb = mu_rgb.detach()
            mu_ir = mu_ir.detach()
        lv_rgb, lv_ir = log_var[:n], log_var[n:]
        kl_sym = sym_kl_diagonal_gaussian(mu_rgb, lv_rgb, mu_ir, lv_ir)
        return kl_sym.mean()


class SigmaAntiCollapseLoss(nn.Module):
    """Hinge penalty if mean(sigma) falls below `floor`.

    Cheap v0 of the Sigma regularizer. Replace with KL-to-modality-prior later
    (see docs/method.md §3 v1).
    """

    def __init__(self, floor: float = 0.05):
        super().__init__()
        self.floor = floor

    def forward(self, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_var)  # (B, D)
        mean_sigma = sigma.mean()
        return F.relu(self.floor - mean_sigma).pow(2)


# ============================================================================
# Class-conditional alignment — directly on raw feat, no dist_head needed.
#
# The original (marginal × parametric × paired) W2 failed because σ described
# the spread of pooled samples per modality and could be driven to 0 jointly
# without harming identity structure. Class-conditional changes the random
# variable: per-ID within-class distribution → σ_k = 0 means that ID's
# features collapse to a point, which fights triplet diversity and CE
# inter-class margin. Shortcut removed.
# ============================================================================


class ClassCondW2Loss(nn.Module):
    """Per-ID diagonal-Gaussian W2 between RGB and IR feature distributions.

    For each identity k present in the batch, estimate
        (μ_v^k, σ_v^k) from K RGB samples,  (μ_t^k, σ_t^k) from K IR samples
    and add ||μ_v^k − μ_t^k||² + ||σ_v^k − σ_t^k||² (per-dim mean, dimension-
    agnostic). σ here is *within-class within-modality* spread; matching it
    across modalities is exactly what cross-modal alignment should do.

    The classic σ-collapse shortcut (σ_v^k = σ_t^k = 0) requires ID k's
    features in BOTH modalities to be a single point — incompatible with
    triplet's pull on intra-class samples and AGW's stochastic batches.

    Requires `feat` and `labels` (the same labels the CE classifier sees).
    Assumes AGW batch layout: feat = [rgb_chunk ; ir_chunk] with paired
    positions (labels[i] == labels[i+N]).
    """

    def __init__(self, std_eps: float = 1e-6, min_samples: int = 2):
        super().__init__()
        self.std_eps = std_eps
        self.min_samples = min_samples  # need ≥2 per modality for std

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_total = feat.shape[0]
        assert n_total % 2 == 0, f"expected even batch, got {n_total}"
        n = n_total // 2
        feat_v, feat_t = feat[:n], feat[n:]
        lbl_v, lbl_t = labels[:n], labels[n:]

        unique_v = torch.unique(lbl_v)
        terms = []
        for k in unique_v:
            mv = lbl_v == k
            mt = lbl_t == k
            if mv.sum().item() < self.min_samples or mt.sum().item() < self.min_samples:
                continue
            fv_k = feat_v[mv]            # (K_v, D)
            ft_k = feat_t[mt]            # (K_t, D)
            mu_v = fv_k.mean(dim=0)
            mu_t = ft_k.mean(dim=0)
            # unbiased=False since K is small (=4); slightly biased but stable
            sigma_v = fv_k.std(dim=0, unbiased=False) + self.std_eps
            sigma_t = ft_k.std(dim=0, unbiased=False) + self.std_eps
            mean_term = (mu_v - mu_t).pow(2).mean()
            std_term = (sigma_v - sigma_t).pow(2).mean()
            terms.append(mean_term + std_term)
        if not terms:
            return feat.new_zeros(())
        return torch.stack(terms).mean()


class ClassCondSWDLoss(nn.Module):
    """Per-ID sliced 1-Wasserstein² (squared) between RGB and IR features.

    For each identity k:
      1. project the K_v RGB and K_t IR samples onto L random unit vectors θ_l
      2. sort each set along each slice
      3. squared difference of sorted projections, mean over slice positions
      4. mean over L slices
    Then mean over identities.

    No Gaussian assumption — captures full distributional shape, not just
    first two moments. Works well for small K (=4) where Gaussian estimates
    are noisy. θ resampled every forward pass for unbiased SWD estimate.

    Requires K_v == K_t (true in AGW's p4n8 batch by construction).
    """

    def __init__(self, n_projections: int = 64):
        super().__init__()
        self.n_projections = n_projections

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_total = feat.shape[0]
        assert n_total % 2 == 0, f"expected even batch, got {n_total}"
        n = n_total // 2
        feat_v, feat_t = feat[:n], feat[n:]
        lbl_v, lbl_t = labels[:n], labels[n:]
        D = feat.shape[1]

        # Random projections — resample per forward for unbiased SWD
        theta = torch.randn(self.n_projections, D, device=feat.device)
        theta = F.normalize(theta, dim=1)                          # (L, D)

        unique_v = torch.unique(lbl_v)
        terms = []
        for k in unique_v:
            mv = lbl_v == k
            mt = lbl_t == k
            kv = mv.sum().item()
            kt = mt.sum().item()
            if kv == 0 or kt == 0:
                continue
            fv_k = feat_v[mv]                                        # (K_v, D)
            ft_k = feat_t[mt]                                        # (K_t, D)
            proj_v = fv_k @ theta.t()                                # (K_v, L)
            proj_t = ft_k @ theta.t()                                # (K_t, L)
            # Equal-K case (typical p4n8): direct sort & subtract
            if kv == kt:
                sv, _ = proj_v.sort(dim=0)
                st, _ = proj_t.sort(dim=0)
                slice_dist = (sv - st).pow(2).mean(dim=0)            # (L,)
            else:
                # Unequal: linearly interpolate to a common quantile grid.
                # Cheap and avoids Sinkhorn for small K.
                sv, _ = proj_v.sort(dim=0)
                st, _ = proj_t.sort(dim=0)
                # quantile positions for each set; resample both to max(K_v,K_t)
                K = max(kv, kt)
                qv = torch.linspace(0, 1, kv, device=feat.device)
                qt = torch.linspace(0, 1, kt, device=feat.device)
                qK = torch.linspace(0, 1, K, device=feat.device)
                # Linear interp along sample-axis, per slice
                sv = _interp_1d(qv, sv, qK)                          # (K, L)
                st = _interp_1d(qt, st, qK)
                slice_dist = (sv - st).pow(2).mean(dim=0)
            terms.append(slice_dist.mean())
        if not terms:
            return feat.new_zeros(())
        return torch.stack(terms).mean()


def _interp_1d(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Linear interpolation along axis 0 of y, with sample positions x (len = y.size(0))
    queried at xq. Used only in unequal-K SWD case.
    """
    # x: (Kx,), y: (Kx, L), xq: (Kq,) → out (Kq, L)
    # Find indices in x for each xq
    idx = torch.searchsorted(x, xq).clamp(1, x.numel() - 1)
    x0 = x[idx - 1]
    x1 = x[idx]
    w = ((xq - x0) / (x1 - x0 + 1e-9)).unsqueeze(1)                # (Kq, 1)
    y0 = y[idx - 1]
    y1 = y[idx]
    return y0 + w * (y1 - y0)
