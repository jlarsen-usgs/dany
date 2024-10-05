import numpy as np


def nash_sutcliffe_efficiency(qsim, qobs, flg=False, nnse=False):
    if flg:
        qsim[qsim == 0] = 1e-06
        qobs[qobs == 0] = 1e-06
        qsim = np.log(qsim)
        qobs = np.log(qobs)
    qsim[np.isinf(qsim)] = np.nan
    qobs[np.isinf(qobs)] = np.nan
    numerator = np.nansum((qobs - qsim) ** 2)
    denominator = np.nansum((qobs - np.nanmean(qobs)) ** 2)
    nse = 1 - (numerator / denominator)
    if nnse:
        nse = 1 / (2 - nse)
    return nse