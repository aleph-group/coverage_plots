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

        c = \bar{x} = \frac{1}{S}\sum_{s=1}^{S} \hat{x}_s,
        \qquad
        r_\alpha = Q_\alpha\bigl(\{\|\hat{x}_s - c\|\}_{s=1}^{S}\bigr),

    where :math:`Q_\alpha` is the empirical :math:`\alpha`-quantile. The
    reference :math:`x` is *covered* at level :math:`\alpha` whenever
    :math:`\|x - c\| \le r_\alpha`. The empirical coverage is the
    fraction of references for which this holds, accumulated across all
    batches passed to :meth:`update`.

    **Choosing the center.** :math:`c` defaults to the posterior mean
    :math:`\bar{x}`, which is the right choice when the samples are draws from
    a posterior and you are asking whether that posterior is calibrated.
    :meth:`update` accepts an explicit ``center`` for the case where they are
    not: if the samples come from a bootstrap around a point estimate
    :math:`\hat{x}` -- each one a reconstruction of re-measured data -- then
    :math:`\bar{x}` is :math:`\hat{x}` plus the reconstructor's bias on its own
    output, an offset that does not shrink with :math:`S`. Centering on
    :math:`\bar{x}` then measures coverage around a point that is never
    reported, and it flatters the method, since the cloud is centered on its
    own mean by construction. Pass ``center=x_hat`` to ask the question that
    matches what is published.

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
    :param Callable distance: callable ``distance(a, b) -> (B,)`` that takes
        two numpy arrays of identical shape ``(B, D)`` (with ``B`` the batch
        of pairs and ``D`` the feature dimension) and returns the per-pair
        distance as a 1-D array of length ``B``. Defaults to the Euclidean
        (L2) norm of ``a - b`` along the last axis. The same distance is
        used for both the pixel-space and (when set) the embedding-space
        coverage.
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

    def __init__(self, embedding=None, distance=None, num_alphas=100, device=None):
        self.embedding = embedding
        self.distance = distance if distance is not None else self._l2_distance
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

    @staticmethod
    def _l2_distance(a, b):
        """Default distance: Euclidean norm of ``a - b`` along the last axis."""
        return np.linalg.norm(a - b, axis=-1)

    def _pairwise(self, samples, center):
        """
        Apply ``self.distance`` between every sample and the corresponding
        center.

        :param samples: array of shape ``(N, S, D)``.
        :param center: array of shape ``(N, D)``.
        :returns: array of shape ``(N, S)`` of distances.
        """
        N, S, D = samples.shape
        a = samples.reshape(N * S, D)
        b = np.broadcast_to(center[:, None, :], (N, S, D)).reshape(N * S, D)
        return np.asarray(self.distance(a, b)).reshape(N, S)

    def _resolve_center(self, x_samples, x_samples_flat, center):
        """
        Pick the center of the credible ball, as an ``(N, D)`` array.

        ``None`` gives the posterior mean :math:`\\bar{x}`, which is the
        default and what every earlier version did unconditionally.

        Anything else is used verbatim, which is what you want when the ball
        should be centered on the estimate you actually report. For an
        iterative or learned reconstructor the sample mean is **not** that
        estimate: the samples are reconstructions of re-measured data, so
        :math:`\\bar{x}` sits at :math:`\\hat{x}` plus the reconstructor's bias
        on its own output, and that offset does not vanish with more samples.
        Measuring coverage around :math:`\\bar{x}` then asks whether the truth
        is near a point nobody publishes, and it flatters the method, because
        the sample cloud is centered on its own mean by construction.

        :param x_samples: the original ``(N, S, C, H, W)`` tensor, for shape
            reporting only.
        :param x_samples_flat: ``(N, S, D)`` flattened samples.
        :param center: ``None``, or a tensor/array broadcastable to the
            reference shape ``(N, C, H, W)`` or already flat ``(N, D)``.
        :returns: ``(N, D)`` array.
        :raises ValueError: if ``center`` does not have ``N`` rows or ``D``
            columns once flattened.
        """
        N, _, D = x_samples_flat.shape
        if center is None:
            return x_samples_flat.mean(axis=1)

        if hasattr(center, "detach"):
            center = self._flatten(center) if center.ndim > 2 else (
                center.detach().cpu().numpy()
            )
        center = np.asarray(center)
        center = center.reshape((center.shape[0], -1))

        if center.shape != (N, D):
            raise ValueError(
                f"center has shape {center.shape} after flattening but the "
                f"samples are {tuple(x_samples.shape)}, which needs ({N}, {D}). "
                "Pass one center per reference, shaped like the reference."
            )
        return center

    def update(self, x_samples, x, center=None):
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
        using the embedded center as the center.

        :param torch.Tensor x_samples: posterior samples of shape
            ``(N, S, C, H, W)`` (``S`` samples per reference).
        :param torch.Tensor x: reference samples of shape ``(N, C, H, W)``.
        :param center: optional center of the ball, one per reference, shaped
            like ``x`` (or already flat, ``(N, D)``). ``None`` -- the default --
            uses the posterior mean :math:`\bar{x}` and reproduces the
            behaviour of every earlier version. Pass the reported estimate
            :math:`\hat{x}` when the question is whether the truth lies near
            *what you publish* rather than near the center of the sample cloud;
            see :meth:`_resolve_center` for why those differ.
        """
        N, S = x_samples.shape[0], x_samples.shape[1]

        # --- Pixel-space (direct) coverage. ---
        x_flat = self._flatten(x)
        x_samples_flat = np.stack(
            [self._flatten(x_samples[:, s]) for s in range(S)], axis=1
        )  # (N, S, D)
        center_img = center
        center = self._resolve_center(x_samples, x_samples_flat, center)
        dist_ref = np.asarray(self.distance(x_flat, center))
        dist_samples = self._pairwise(x_samples_flat, center)
        # (num_alphas, N) — linearly-interpolated empirical alpha-quantiles
        # per reference; well-defined for any S, including alpha=0/alpha=1.
        quantiles = np.quantile(dist_samples, self.alphas, axis=1)
        self.sums += np.sum(dist_ref[None, :] <= quantiles, axis=1)

        # --- Optional embedding-space coverage. ---
        if self.embedding is not None:
            # The embedding is a map on IMAGES, so the center has to be
            # embedded as an image rather than reused from the flat pixel-space
            # center -- phi(xbar) is not the mean of phi(x_s). When a center was
            # supplied it must therefore arrive in image form; a caller who
            # passed a pre-flattened (N, D) center has nothing to embed.
            if center_img is None:
                x_center_img = x_samples.mean(dim=1)
            elif hasattr(center_img, "ndim") and center_img.ndim == x.ndim:
                x_center_img = center_img
            else:
                raise ValueError(
                    "embedding-space coverage needs `center` shaped like `x` "
                    f"({tuple(x.shape)}) so it can be embedded; got "
                    f"{tuple(np.shape(center_img))}."
                )
            x_meanemb = self._embed(x_center_img)
            x_emb = self._embed(x)
            x_samples_emb = np.stack(
                [self._embed(x_samples[:, s]) for s in range(S)], axis=1
            )
            dist_ref_e = np.asarray(self.distance(x_emb, x_meanemb))
            dist_samples_e = self._pairwise(x_samples_emb, x_meanemb)
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
