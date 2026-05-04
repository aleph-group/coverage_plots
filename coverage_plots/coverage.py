import numpy as np


class Coverage:
    r"""
    Empirical coverage metric for posterior samples in inverse problems.

    Given a reference sample :math:`x` (e.g. a ground-truth image) and a set of
    posterior samples :math:`\{\hat{x}_s\}_{s=1}^S` produced by a stochastic
    reconstruction algorithm, this metric measures whether :math:`x` lies
    within the bulk of the posterior. For a coverage level
    :math:`\alpha \in [0, 1]`, define

    .. math::

        \bar{x} = \frac{1}{S}\sum_{s=1}^{S} \hat{x}_s,
        \qquad
        r_\alpha = Q_\alpha\bigl(\{\|\hat{x}_s - \bar{x}\|\}_{s=1}^{S}\bigr),

    where :math:`Q_\alpha` is the empirical :math:`\alpha`-quantile. The
    reference :math:`x` is *covered* at level :math:`\alpha` whenever
    :math:`\|x - \bar{x}\| \le r_\alpha`. The empirical coverage is the
    fraction of references for which this holds, accumulated across all
    batches passed to :meth:`update`.

    A well-calibrated posterior produces a coverage curve close to the diagonal
    :math:`\mathrm{coverage}(\alpha) = \alpha`; curves below the diagonal
    indicate over-confident posteriors, curves above indicate under-confident
    ones.

    The pixel-space (direct) coverage curve is *always* computed. When an
    optional ``embedding`` callable :math:`\phi` is provided, an additional
    embedding-space curve is computed in parallel using the *feature of the
    pixel-space mean*, :math:`\phi(\bar{x})`, as the center; both curves are
    returned by :meth:`compute`. The embedding-space curve is ``None`` when
    no embedding is set.

    :param Callable embedding: optional callable mapping a batch tensor of
        shape ``(B, C, H, W)`` to a feature tensor of shape ``(B, D)``. The
        output may be a ``torch.Tensor`` (it is detached and moved to CPU) or
        a ``numpy.ndarray``. If ``None``, only the pixel-space curve is
        computed.
    :param int num_alphas: number of evenly-spaced quantile levels in
        :math:`[0, 1]` at which the coverage curve is evaluated.
    :param torch.device device: device on which the embedding is expected to
        run. Stored for convenience; the class itself does no device handling.

    |sep|

    :Examples:

        >>> import torch
        >>> from coverage_plots import Coverage
        >>> cov = Coverage(num_alphas=50)
        >>> x = torch.randn(8, 1, 16, 16)              # references
        >>> x_samples = x.unsqueeze(1) + 0.1 * torch.randn(8, 32, 1, 16, 16)
        >>> cov.update(x_samples, x)
        >>> alphas, coverage, coverage_emb = cov.compute()  # coverage_emb is None
    """

    def __init__(self, embedding=None, num_alphas=100, device=None):
        self.embedding = embedding
        self.device = device
        self.num_alphas = num_alphas
        self.reset()

    def reset(self):
        """
        Reset all accumulated statistics.

        Clears the running sums and sample counter so the instance can be
        re-used to compute a fresh coverage curve. Called automatically by
        :meth:`__init__`.
        """
        self.alphas = np.linspace(0, 1, self.num_alphas)
        self.sums = np.zeros(self.num_alphas)
        self.sums_emb = (
            np.zeros(self.num_alphas) if self.embedding is not None else None
        )
        self.samples = 0

    def _flatten(self, x):
        """Flatten ``(B, C, H, W)`` -> ``(B, C*H*W)`` numpy array."""
        return x.detach().cpu().numpy().reshape((x.shape[0], -1))

    def _embed(self, x):
        """Apply ``self.embedding`` and return a ``(B, D)`` numpy array."""
        return self.embedding(x).detach().cpu().numpy()

    def update(self, x_samples, x):
        r"""
        Accumulate coverage statistics for a new batch of references.

        For each of the ``N`` reference samples in ``x``, the corresponding
        ``S`` posterior samples in ``x_samples`` are used to compute the
        empirical :math:`\alpha`-quantile of within-cluster distances at every
        :math:`\alpha \in` ``self.alphas``, and the running coverage counters
        are incremented whenever the reference falls inside that radius. The
        batch size ``N`` is added to the internal sample counter.

        Pixel-space distances are always computed. If ``self.embedding`` is
        set, the same accounting is also performed in the embedding space,
        using :math:`\phi(\bar{x})` as the center.

        :param torch.Tensor x_samples: posterior samples of shape
            ``(N, S, C, H, W)`` (``S`` samples per reference).
        :param torch.Tensor x: reference samples of shape ``(N, C, H, W)``.
        """
        N, S = x_samples.shape[0], x_samples.shape[1]

        # --- Pixel-space (direct) coverage. ---
        x_flat = self._flatten(x)
        x_samples_flat = np.stack(
            [self._flatten(x_samples[:, s]) for s in range(S)], axis=1
        )  # (N, S, D)
        center = x_samples_flat.mean(axis=1)
        dist_ref = np.linalg.norm(x_flat - center, axis=-1)
        dist_samples = np.linalg.norm(x_samples_flat - center[:, None, :], axis=-1)
        # (num_alphas, N) — linearly-interpolated empirical alpha-quantiles
        # per reference; well-defined for any S, including alpha=0/alpha=1.
        quantiles = np.quantile(dist_samples, self.alphas, axis=1)
        self.sums += np.sum(dist_ref[None, :] <= quantiles, axis=1)

        # --- Optional embedding-space coverage. ---
        if self.embedding is not None:
            x_mean_img = x_samples.mean(dim=1)
            x_meanemb = self._embed(x_mean_img)
            x_emb = self._embed(x)
            x_samples_emb = np.stack(
                [self._embed(x_samples[:, s]) for s in range(S)], axis=1
            )
            dist_ref_e = np.linalg.norm(x_emb - x_meanemb, axis=-1)
            dist_samples_e = np.linalg.norm(
                x_samples_emb - x_meanemb[:, None, :], axis=-1
            )
            quantiles_e = np.quantile(dist_samples_e, self.alphas, axis=1)
            self.sums_emb += np.sum(dist_ref_e[None, :] <= quantiles_e, axis=1)

        self.samples += N

    def compute(self, csv_path=None):
        r"""
        Finalise and return the empirical coverage curves.

        Divides the accumulated counters by the total number of references
        seen so far to produce the pixel-space coverage curve over
        ``self.alphas`` and, when ``self.embedding`` is set, an additional
        embedding-space curve. If no batches have been processed yet, the raw
        zero-valued counters are returned unchanged.

        :param str | os.PathLike csv_path: optional path to a CSV file where
            the curves are saved with columns ``alpha``, ``coverage`` and
            (when an embedding is set) ``coverage_emb``. Requires ``pandas``.
        :returns: tuple ``(alphas, coverage, coverage_emb)`` of
            ``numpy.ndarray`` of shape ``(num_alphas,)``. ``coverage_emb`` is
            ``None`` when no embedding was provided.
        """
        if self.samples == 0:
            return self.alphas, self.sums, self.sums_emb

        cov = self.sums / self.samples
        cov_emb = self.sums_emb / self.samples if self.sums_emb is not None else None

        # Endpoints are known analytically: a zero-radius ball covers nothing
        # and a full-radius (alpha = 1) ball covers every reference.
        if self.alphas[0] == 0.0:
            cov[0] = 0.0
            if cov_emb is not None:
                cov_emb[0] = 0.0
        if self.alphas[-1] == 1.0:
            cov[-1] = 1.0
            if cov_emb is not None:
                cov_emb[-1] = 1.0

        if csv_path is not None:
            from pandas import DataFrame

            data = {"alpha": self.alphas, "coverage": cov}
            if cov_emb is not None:
                data["coverage_emb"] = cov_emb
            DataFrame(data).to_csv(csv_path, index=False)

        return self.alphas, cov, cov_emb
