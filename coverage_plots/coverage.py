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

    Two curves are tracked in parallel and both returned by :meth:`compute`:

    * ``coverage`` uses the mean of the *features* of the samples,
      :math:`\bar{\phi} = \tfrac{1}{S}\sum_s \phi(\hat{x}_s)`, as the center.
    * ``coverage_meanemb`` uses the *feature of the pixel-space mean*,
      :math:`\phi(\bar{x})`, as the center. The two coincide when ``embedding``
      is ``None`` (pixel-space) or when :math:`\phi` is linear.

    Distances are computed in feature space defined by the optional
    ``embedding`` callable (e.g. an encoder). When ``embedding`` is ``None``,
    samples are simply flattened and distances are Euclidean in pixel space.

    :param Callable embedding: optional callable mapping a batch tensor of
        shape ``(B, C, H, W)`` to a feature tensor of shape ``(B, D)``. The
        output may be a ``torch.Tensor`` (it is detached and moved to CPU) or
        a ``numpy.ndarray``. If ``None``, the raw samples are flattened to
        ``(B, C*H*W)`` and used as features.
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
        >>> alphas, coverage, coverage_meanemb = cov.compute()
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
        self.sums_meanemb = np.zeros(self.num_alphas)
        self.samples = 0

    def _to_features(self, x):
        """
        Map a batch ``x`` of shape ``(B, C, H, W)`` to a feature array of
        shape ``(B, D)`` as a numpy array, applying ``self.embedding`` if set,
        otherwise just flattening the spatial/channel dimensions.
        """
        if self.embedding is not None:
            return self.embedding(x).detach().cpu().numpy()
        return x.detach().cpu().numpy().reshape((x.shape[0], -1))

    def update(self, x_samples, x):
        r"""
        Accumulate coverage statistics for a new batch of references.

        For each of the ``N`` reference samples in ``x``, the corresponding
        ``S`` posterior samples in ``x_samples`` are embedded, the empirical
        :math:`\alpha`-quantile of within-cluster distances is computed for
        every :math:`\alpha \in` ``self.alphas``, and the running coverage
        counters are incremented whenever the reference falls inside that
        radius. The batch size ``N`` is added to the internal sample counter.

        :param torch.Tensor x_samples: posterior samples of shape
            ``(N, S, C, H, W)`` (``S`` samples per reference).
        :param torch.Tensor x: reference samples of shape ``(N, C, H, W)``.
        """
        N, S = x_samples.shape[0], x_samples.shape[1]

        # Mean image across samples (in image space), then embed it.
        x_mean_img = x_samples.mean(dim=1)
        x_meanemb = self._to_features(x_mean_img)

        # Embed the reference samples.
        x_emb = self._to_features(x)

        # Embed each sample (loop over the sample dimension).
        x_samples_emb = np.stack(
            [self._to_features(x_samples[:, s]) for s in range(S)],
            axis=1,
        )  # shape (N, S, D)

        # Mean of embedded samples.
        x_mean_emb = x_samples_emb.mean(axis=1)

        dist_mean = np.linalg.norm(x_emb - x_mean_emb, axis=-1)
        dist_meanemb = np.linalg.norm(x_emb - x_meanemb, axis=-1)

        dist_samples = np.linalg.norm(x_samples_emb - x_mean_emb[:, None, :], axis=-1)
        dist_samples = np.sort(dist_samples, axis=1)

        dist_samples_meanemb = np.linalg.norm(
            x_samples_emb - x_meanemb[:, None, :], axis=-1
        )
        dist_samples_meanemb = np.sort(dist_samples_meanemb, axis=1)

        for i, alpha in enumerate(self.alphas):
            idx = int(alpha * (S - 1))
            self.sums[i] += np.sum(dist_mean <= dist_samples[:, idx])
            self.sums_meanemb[i] += np.sum(dist_meanemb <= dist_samples_meanemb[:, idx])

        self.samples += N

    def compute(self, csv_path=None):
        r"""
        Finalise and return the empirical coverage curves.

        Divides the accumulated counters by the total number of references
        seen so far to produce two coverage curves over ``self.alphas``
        (see the class docstring for the difference between them). If no
        batches have been processed yet, the raw zero-valued counters are
        returned unchanged.

        :param str | os.PathLike csv_path: optional path to a CSV file where
            the curves are saved with columns ``alpha``, ``coverage`` and
            ``coverage_meanemb``. Requires ``pandas`` to be installed.
        :returns: tuple ``(alphas, coverage, coverage_meanemb)`` of
            ``numpy.ndarray`` of shape ``(num_alphas,)``.
        """
        if self.samples == 0:
            return self.alphas, self.sums, self.sums_meanemb

        cov = self.sums / self.samples
        cov_meanemb = self.sums_meanemb / self.samples

        if csv_path is not None:
            from pandas import DataFrame

            DataFrame(
                {
                    "alpha": self.alphas,
                    "coverage": cov,
                    "coverage_meanemb": cov_meanemb,
                }
            ).to_csv(csv_path, index=False)

        return self.alphas, cov, cov_meanemb
