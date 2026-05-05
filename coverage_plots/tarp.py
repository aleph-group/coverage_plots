r"""
TARP coverage diagnostic for posterior samples in inverse problems.

Reference: Lemos, Coogan et al. 2023, *Sampling-Based Accuracy Testing of
Posterior Estimators for General Inference* (https://arxiv.org/abs/2302.03026).

The implementation here is a numpy port of the function ``run_tarp`` from the
``sbi`` package, adapted to the conventions of :class:`coverage_plots.Coverage`:
references are images of shape ``(N, C, H, W)`` and posterior samples have
shape ``(N, S, C, H, W)``. An optional ``embedding`` callable can be used to
run the diagnostic in a feature space, and the underlying distance is
configurable (defaults to L2).
"""

from __future__ import annotations

import numpy as np


def _flatten(x):
    """``(B, C, H, W)`` torch/numpy -> ``(B, C*H*W)`` numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.reshape((x.shape[0], -1))


def _embed(embedding, x):
    """Apply ``embedding`` to a torch tensor and return a ``(B, D)`` numpy array."""
    out = embedding(x)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out)


def _l2_distance(a, b):
    """Default distance: Euclidean norm of ``a - b`` along the last axis."""
    return np.linalg.norm(a - b, axis=-1)


def _pairwise(distance, samples, center):
    """
    Apply ``distance`` between every sample and the corresponding center.

    :param samples: array of shape ``(N, S, D)``.
    :param center: array of shape ``(N, D)``.
    :returns: array of shape ``(N, S)``.
    """
    N, S, D = samples.shape
    a = samples.reshape(N * S, D)
    b = np.broadcast_to(center[:, None, :], (N, S, D)).reshape(N * S, D)
    return np.asarray(distance(a, b)).reshape(N, S)


def run_tarp(
    x_samples,
    x,
    embedding=None,
    distance=None,
    references=None,
    num_bins=None,
    z_score=True,
    rng=None,
):
    r"""
    Estimate coverage of posterior samples with the TARP method.

    Implements algorithm 2 of Lemos et al. 2023. For each reference :math:`x_n`
    a uniform reference point :math:`\theta^\star_n` is drawn over the bounding
    box of the references, then the empirical credibility level

    .. math::

        f_n = \frac{1}{S}\sum_{s=1}^{S}
              \mathbf{1}\bigl[d(\theta^\star_n, \hat{x}_{n,s}) <
                              d(\theta^\star_n, x_n)\bigr]

    is collected over all references. The expected coverage probability
    (``ecp``) is the empirical CDF of :math:`\{f_n\}_n`, evaluated on a grid
    of :math:`\alpha` values.

    A well-calibrated posterior produces ``ecp`` close to the diagonal
    :math:`\mathrm{ecp}(\alpha) = \alpha`.

    :param torch.Tensor x_samples: posterior samples of shape
        ``(N, S, C, H, W)`` (``S`` samples per reference).
    :param torch.Tensor x: references of shape ``(N, C, H, W)``.
    :param Callable embedding: optional callable mapping a batch tensor of
        shape ``(B, C, H, W)`` to features of shape ``(B, D)``. If ``None``,
        the diagnostic is computed in flattened pixel space.
    :param Callable distance: callable ``distance(a, b) -> (B,)`` taking two
        numpy arrays of identical shape ``(B, D)``. Defaults to L2.
    :param numpy.ndarray references: optional array of shape ``(N, D)`` of
        custom reference points. If ``None``, references are drawn from a
        uniform distribution over the per-dimension bounding box of ``x``.
    :param int num_bins: number of points in the credibility grid on which the
        empirical CDF of the credibility values is evaluated. Defaults to
        ``max(N, 100)``, where ``N`` is the number of references.
    :param bool z_score: whether to min-max normalize each dimension to
        ``[0, 1]`` before computing distances. Recommended for image inputs
        where pixel scales are heterogeneous.
    :param numpy.random.Generator rng: optional random generator used to draw
        the uniform reference points. Defaults to ``numpy.random.default_rng()``.

    :returns: tuple ``(alpha, ecp)`` of ``numpy.ndarray`` of shape
        ``(num_bins,)``. ``alpha`` is the credibility grid spanning ``[0, 1]``
        and ``ecp`` is the corresponding expected coverage probability.
    """
    if distance is None:
        distance = _l2_distance
    if rng is None:
        rng = np.random.default_rng()

    # --- Flatten / embed references and samples to (N, D) and (N, S, D). ---
    if embedding is None:
        x_flat = _flatten(x)
        S = x_samples.shape[1]
        x_samples_flat = np.stack([_flatten(x_samples[:, s]) for s in range(S)], axis=1)
    else:
        x_flat = _embed(embedding, x)
        S = x_samples.shape[1]
        x_samples_flat = np.stack(
            [_embed(embedding, x_samples[:, s]) for s in range(S)], axis=1
        )

    N, _, D = x_samples_flat.shape

    if num_bins is None:
        num_bins = max(N, 100)

    # --- Reference points: uniform over the bounding box of x. ---
    if references is None:
        lo = x_flat.min(axis=0)
        hi = x_flat.max(axis=0)
        references = rng.uniform(low=lo, high=hi, size=(N, D))
    else:
        references = np.asarray(references)
        assert (
            references.shape == x_flat.shape
        ), f"references must have shape {x_flat.shape}, got {references.shape}"

    # --- Optional per-dimension min-max normalization. ---
    if z_score:
        lo = x_flat.min(axis=0, keepdims=True)
        hi = x_flat.max(axis=0, keepdims=True)
        scale = hi - lo + 1e-10
        x_flat = (x_flat - lo) / scale
        x_samples_flat = (x_samples_flat - lo[None, :, :]) / scale[None, :, :]
        references = (references - lo) / scale

    # --- Distances and per-reference credibility values. ---
    sample_dists = _pairwise(distance, x_samples_flat, references)  # (N, S)
    theta_dists = np.asarray(distance(x_flat, references))  # (N,)
    coverage_values = np.mean(sample_dists < theta_dists[:, None], axis=1)  # (N,)

    # --- Empirical CDF of the credibility values on a fixed [0, 1] grid. ---
    # ecp(alpha) = (1/N) * #{ n : f_n <= alpha }. This matches equation 4 of
    # the paper, is bin-choice-free, and always spans the full [0, 1] range
    # regardless of how many references are available.
    alpha = np.linspace(0.0, 1.0, num_bins)
    sorted_cv = np.sort(coverage_values)
    ecp = np.searchsorted(sorted_cv, alpha, side="right") / N
    # Pin endpoints: by definition no posterior credibility ball of radius 0
    # covers the truth, and the radius-1 ball always does.
    ecp[0] = 0.0
    ecp[-1] = 1.0
    return alpha, ecp
