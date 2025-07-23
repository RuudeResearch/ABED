"""
    @author: Magnus Ruud Kjær
    updated from: https://doi.org/10.1016/j.jneumeth.2019.03.017
"""

import torch


def decode(localization, localizations_default, variance=0.1):
    """decode

    Parameters
    ----------
    localization : array-like, shape (n_loc, 2)
        The array of predicted and encoded localizations
    localizations_default : array-like, shape (n_loc, 2)
        The array of default localization
    variance : float, optional (default=0.1)
        allowed variance in position estimation

    Returns:
    --------
    localization_decoded : array-like, shape (n_loc, 2)
        The array of decoded localizations
    """

    center_encoded, width_encoded = localization[:, 0], localization[:, 1]

    x_plus_y = (center_encoded * localizations_default[:, 1] * variance +
                localizations_default[:, 0]) * 2

    y_minus_x = torch.exp(width_encoded * variance) * \
        localizations_default[:, 1]

    x = (x_plus_y - y_minus_x) / 2
    y = (x_plus_y + y_minus_x) / 2

    localization_decoded = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], 1)

    return localization_decoded
