
"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""

import torch


def encode(localization_match, localizations_default, variance=0.1):
    """localization_match are converted relatively to their default location

    Parameters:
    -----------
    localization_match : array-like, shape (batch_size, n_loc, 2)
        Array containing the ground truth matched localization
        (representation x y)
    localizations_default : array-like, shape (n_loc, 2)
        Array of default localization
    variance : float, optional (default=0.1)
        The allowed variance

    Returns:
    --------
    localization_target : array-like, shape (batch, n_loc, 2)
        Array of encoded localizations
    """

    center = (localization_match[:, 0] + localization_match[:, 1]) / 2 -\
        localizations_default[:, 0]
    center = center / (variance * localizations_default[:, 1])

    width = torch.log((localization_match[:, 1] - localization_match[:, 0]) /
                      localizations_default[:, 1]) / variance

    localization_target = torch.cat(
        [center.unsqueeze(1), width.unsqueeze(1)], 1)

    return localization_target
