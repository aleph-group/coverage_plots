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

Two entry points:

:class:`TARP`
    Accumulator, mirroring :class:`~coverage_plots.Coverage`. Consumes one batch
    at a time, so peak memory is one batch rather than the whole sample set.
    Required at image sizes where the full array does not fit: at
    ``N=200, S=100`` on ``360x360`` float32 the monolithic form needs 10.4 GB,
    and makes a second copy while flattening.
:func:`run_tarp`
    The original one-shot function, kept for the existing API. A thin wrapper
    over :class:`TARP`.

Streaming is nearly free here because the per-reference credibility value
:math:`f_n` is independent across references. The only genuinely global
quantities are the per-dimension bounding box that :math:`\theta^\star` is drawn
from and the ``z_score`` min/max — the same statistic — which is why
:meth:`TARP.fit_bounds` exists and must see the references before
:meth:`TARP.update` is called.
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


class TARP:
    r"""
    Streaming TARP coverage, accumulated over batches of references.

    Usage mirrors :class:`~coverage_plots.Coverage`, with one extra step: the
    bounding box must be fitted first, because :math:`\theta^\star` is drawn from
    it and it is a property of the reference set as a whole.

    .. code-block:: python

        tarp = TARP(rng=np.random.default_rng(0))
        tarp.fit_bounds(all_references)          # cheap: (N, D), no samples
        for x, x_samples in batches:
            tarp.update(x_samples, x)
        alpha, ecp = tarp.compute()

    :param int num_bins: number of points in the credibility grid. Defaults to
        ``max(N, 100)`` once the total reference count ``N`` is known.
    :param Callable embedding: optional callable mapping ``(B, C, H, W)`` to
        features ``(B, D)``. If ``None``, the diagnostic runs in flattened pixel
        space.

        .. note::
            In high dimension the diagnostic can lose its discriminating power:
            :math:`\theta^\star` is drawn uniformly over a per-dimension bounding
            box, and in (say) 129 600 dimensions such a draw is white noise, far
            from every reference and every sample. The
            :math:`\{f_n\}` distribution then concentrates and the curve stops
            separating models. This is a property of the method, not of this
            implementation; the fix is to pass an ``embedding`` and run the test
            on a low-dimensional summary. Inspect
            :attr:`credibility_values` before trusting a pixel-space curve.
    :param Callable distance: callable ``distance(a, b) -> (B,)`` on two arrays
        of identical shape ``(B, D)``. Defaults to L2.
    :param bool z_score: min-max normalise each dimension to ``[0, 1]`` using the
        fitted bounds before computing distances. Recommended for images, where
        pixel scales are heterogeneous.
    :param numpy.random.Generator rng: generator for the reference points. Pass a
        seeded one for reproducible curves; the default is unseeded, which makes
        the result vary between runs.
    """

    def __init__(
        self,
        num_bins=None,
        embedding=None,
        distance=None,
        z_score=True,
        rng=None,
    ):
        self.num_bins = num_bins
        self.embedding = embedding
        self.distance = distance if distance is not None else _l2_distance
        self.z_score = z_score
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        """Drop accumulated credibility values and fitted bounds."""
        self.credibility_values = []
        self._lo = None
        self._hi = None

    # -- bounds -----------------------------------------------------------

    def _features(self, x):
        return _flatten(x) if self.embedding is None else _embed(self.embedding, x)

    def fit_bounds(self, x):
        r"""
        Accumulate the per-dimension bounding box of the references.

        May be called repeatedly to build the box incrementally; the running
        min/max is kept. It must see the *same* reference set that
        :meth:`update` is later called with, since :math:`\theta^\star` is drawn
        from this box and the ``z_score`` normalisation uses it.

        Only the references are needed, never the samples, so this pass is cheap:
        ``(N, D)`` rather than ``(N, S, D)``.

        :param torch.Tensor x: references of shape ``(N, C, H, W)``.
        """
        feats = self._features(x)
        lo, hi = feats.min(axis=0), feats.max(axis=0)
        self._lo = lo if self._lo is None else np.minimum(self._lo, lo)
        self._hi = hi if self._hi is None else np.maximum(self._hi, hi)

    # -- accumulation -----------------------------------------------------

    def update(self, x_samples, x):
        r"""
        Accumulate credibility values for a batch of references.

        For each reference :math:`x_n` a uniform point :math:`\theta^\star_n` is
        drawn over the fitted bounding box and

        .. math::

            f_n = \frac{1}{S}\sum_{s=1}^{S}
                  \mathbf{1}\bigl[d(\theta^\star_n, \hat{x}_{n,s}) <
                                  d(\theta^\star_n, x_n)\bigr]

        is recorded. Nothing else is retained from the batch.

        :param torch.Tensor x_samples: posterior samples ``(N, S, C, H, W)``.
        :param torch.Tensor x: references ``(N, C, H, W)``.
        :raises RuntimeError: if :meth:`fit_bounds` has not been called.
        """
        if self._lo is None:
            raise RuntimeError(
                "call fit_bounds(references) before update(): the TARP reference "
                "points are drawn from the bounding box of the whole reference "
                "set, which cannot be known from a single batch."
            )

        x_flat = self._features(x)
        S = x_samples.shape[1]
        x_samples_flat = np.stack(
            [self._features(x_samples[:, s]) for s in range(S)], axis=1
        )
        N, _, D = x_samples_flat.shape

        references = self.rng.uniform(low=self._lo, high=self._hi, size=(N, D))

        if self.z_score:
            scale = self._hi - self._lo + 1e-10
            x_flat = (x_flat - self._lo) / scale
            x_samples_flat = (x_samples_flat - self._lo[None, None, :]) / scale[
                None, None, :
            ]
            references = (references - self._lo) / scale

        sample_dists = _pairwise(self.distance, x_samples_flat, references)  # (N, S)
        theta_dists = np.asarray(self.distance(x_flat, references))  # (N,)
        batch_cv = np.mean(sample_dists < theta_dists[:, None], axis=1)  # (N,)
        self.credibility_values.append(batch_cv)

    # -- finalise ---------------------------------------------------------

    def compute(self, csv_path=None):
        r"""
        Finalise and return ``(alpha, ecp)``.

        The expected coverage probability is the empirical CDF of the
        accumulated :math:`\{f_n\}` on a fixed grid in ``[0, 1]``. This matches
        equation 4 of the paper, is bin-choice-free, and always spans the full
        range regardless of how many references were seen.

        :param str | os.PathLike csv_path: optional path to write the curve to,
            with columns ``alpha`` and ``ecp``. Requires ``pandas``.
        :returns: tuple ``(alpha, ecp)`` of ``numpy.ndarray``, shape
            ``(num_bins,)``.
        """
        if not self.credibility_values:
            raise RuntimeError("no batches accumulated; call update() first")

        coverage_values = np.concatenate(self.credibility_values)
        N = coverage_values.size
        num_bins = self.num_bins if self.num_bins is not None else max(N, 100)

        alpha = np.linspace(0.0, 1.0, num_bins)
        sorted_cv = np.sort(coverage_values)
        ecp = np.searchsorted(sorted_cv, alpha, side="right") / N
        # Pin endpoints: by definition no posterior credibility ball of radius 0
        # covers the truth, and the radius-1 ball always does.
        ecp[0] = 0.0
        ecp[-1] = 1.0

        if csv_path is not None:
            import pandas as pd

            pd.DataFrame({"alpha": alpha, "ecp": ecp}).to_csv(csv_path, index=False)

        return alpha, ecp


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

    .. note::
        This holds every sample in memory at once. For large images or many
        references use :class:`TARP`, which accumulates batch by batch.

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
    tarp = TARP(
        num_bins=num_bins,
        embedding=embedding,
        distance=distance,
        z_score=z_score,
        rng=rng,
    )
    tarp.fit_bounds(x)

    if references is not None:
        # Custom reference points bypass the bounding-box draw entirely, so feed
        # them through a one-shot pass rather than the accumulator's sampler.
        references = np.asarray(references)
        x_flat = tarp._features(x)
        if references.shape != x_flat.shape:
            raise AssertionError(
                f"references must have shape {x_flat.shape}, got {references.shape}"
            )

        class _Fixed:
            def __init__(self, values):
                self.values = values

            def uniform(self, low, high, size):
                return self.values

        tarp.rng = _Fixed(references)

    tarp.update(x_samples, x)
    return tarp.compute()
