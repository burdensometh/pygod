# -*- coding: utf-8 -*-
"""Train-fitted feature imputation and scaling for missing values."""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch


class FeatureImputer:
    """
    Imputation and scaling for node features with missing values
    (``NaN``), with statistics fitted on the training data only.

    Missing entries are filled with the per-feature mean computed over
    the *observed* training entries, and features are standardized with
    the per-feature mean/std computed over the observed training
    entries. The fitted state (``mean_``, ``std_``) is stored on the
    instance and reused by :meth:`transform`, so validation/test data
    never influences the training statistics.

    In the transformed space, imputed entries are exactly ``0`` (the
    training mean), i.e. a neutral value rather than a fabricated
    observation. The observation mask returned by :meth:`transform`
    marks which entries were truly observed, so downstream
    reconstruction losses can ignore imputed dimensions.

    Explicit policies for degenerate cases:

    - A feature never observed in the training data gets ``mean_=0``
      and ``std_=1`` (its imputed value is ``0``).
    - A feature with zero observed variance gets ``std_=1``.
    - A node with all features missing is imputed to the training
      feature means (``0`` after scaling) and its mask row is all
      ``0``; how detectors score such nodes is defined by the
      reconstruction loss (see
      :func:`pygod.nn.functional.double_recon_loss`).

    Parameters
    ----------
    scale : bool, optional
        Whether to standardize features with the fitted training
        mean/std. Default: ``True``.

    Attributes
    ----------
    mean_ : torch.Tensor
        Per-feature mean over the observed training entries, of shape
        :math:`F`. ``0`` for features never observed in training.
    std_ : torch.Tensor
        Per-feature standard deviation over the observed training
        entries, of shape :math:`F`. ``1`` for features never observed
        or with zero observed variance.
    n_obs_ : torch.Tensor
        Number of observed training entries per feature, of shape
        :math:`F`.

    Examples
    --------
    >>> from pygod.utils import FeatureImputer
    >>> imputer = FeatureImputer()
    >>> x_train_filled, train_mask = imputer.fit_transform(x_train)
    >>> x_test_filled, test_mask = imputer.transform(x_test)  # refits nothing
    """

    def __init__(self, scale=True):
        self.scale = scale
        self.mean_ = None
        self.std_ = None
        self.n_obs_ = None

    def fit(self, x):
        """Fit the imputation and scaling statistics on training data.

        Parameters
        ----------
        x : torch.Tensor
            Training node feature matrix of shape :math:`N \\times F`,
            with missing values encoded as ``NaN``.

        Returns
        -------
        self : FeatureImputer
            The fitted imputer.
        """
        x = self._to_float(x)
        obs = ~torch.isnan(x)
        n_obs = obs.sum(dim=0)

        # nansum treats NaN as 0, so the statistics are computed over
        # the observed entries only
        mean = torch.nansum(x, dim=0) / n_obs.clamp(min=1)
        # explicit policy: features never observed get mean 0
        mean = torch.where(n_obs > 0, mean, torch.zeros_like(mean))

        if self.scale:
            var = torch.nansum((x - mean) ** 2, dim=0) / n_obs.clamp(min=1)
            std = torch.sqrt(var)
            # explicit policy: never-observed or constant features get std 1
            std = torch.where((n_obs > 0) & (std > 0), std,
                              torch.ones_like(std))
        else:
            std = torch.ones_like(mean)

        self.mean_ = mean
        self.std_ = std
        self.n_obs_ = n_obs
        return self

    def transform(self, x):
        """Impute and scale features with the fitted training state.

        This method never updates the fitted statistics, so applying it
        to validation/test data cannot leak information into the
        training state.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape :math:`N \\times F`, with
            missing values encoded as ``NaN``.

        Returns
        -------
        x_out : torch.Tensor
            The imputed and scaled feature matrix of shape
            :math:`N \\times F`. Contains no ``NaN``; previously
            missing entries are ``0`` (the training mean) when
            ``scale=True``.
        mask : torch.Tensor
            The observation mask of shape :math:`N \\times F`, with
            ``1`` for observed and ``0`` for missing entries, in the
            same dtype as ``x_out``.
        """
        assert self.mean_ is not None and self.std_ is not None, \
            "The imputer is not fitted yet"
        x = self._to_float(x)
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError("Expected %d features, got %d"
                             % (self.mean_.shape[0], x.shape[1]))

        mean = self.mean_.to(x.device)
        std = self.std_.to(x.device)

        mask = ~torch.isnan(x)
        x_filled = torch.where(mask, x, mean)
        x_out = (x_filled - mean) / std
        return x_out, mask.to(x_out.dtype)

    def fit_transform(self, x):
        """Fit on the training data and transform it.

        Parameters
        ----------
        x : torch.Tensor
            Training node feature matrix of shape :math:`N \\times F`,
            with missing values encoded as ``NaN``.

        Returns
        -------
        x_out : torch.Tensor
            The imputed and scaled feature matrix.
        mask : torch.Tensor
            The observation mask (``1`` observed, ``0`` missing).
        """
        return self.fit(x).transform(x)

    @staticmethod
    def _to_float(x):
        if not torch.is_floating_point(x):
            x = x.float()
        return x
