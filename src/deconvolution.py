import numpy as np
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error as mse


class Deconvolution:
    def __init__(self, nu_list=[0.25, 0.5, 0.75], verbose=True):
        """
        Initializes three linear NuSVR machines, with nu in [0.25, 0.5, 0.75].
        The best model is saved where 'best' is measured as the model that
        produces the lowest r.m.s. error between y and X @ coefs.
        """
        self.verbose = verbose
        if np.min(nu_list) <= 0 or np.max(nu_list) >= 1:
            raise ValueError("Nu must be between 0 and 1.")
        self.nusvr_list = [NuSVR(kernel="linear", nu=nu) for nu in nu_list]

    def fit_predict(self, X, y):
        # CIBERSORT normalizes over the entire data
        X = (X - X.mean()) / np.sqrt(X.var())
        y = (y - y.mean()) / np.sqrt(y.var())
        best_rms = np.Inf
        best_nusvr = None

        for nusvr in self.nusvr_list:
            if self.verbose:
                print(f"Fitting NuSVR with nu={nusvr.nu}.")
            nusvr.fit(X, y)

            coefs = nusvr.coef_.flatten()
            rms_error = np.sqrt(mse(y, X @ coefs[:, None]))
            if self.verbose:
                print(f"RMS: {rms_error}")
            if rms_error < best_rms:
                best_rms = rms_error
                best_nusvr = nusvr

        if self.verbose:
            print(f"Selected best model with nu={best_nusvr.nu}.")
        self.best_nusvr = best_nusvr
        return _correct_coefs(best_nusvr.coef_)


def _correct_coefs(coefs):
    """
    Given a vector of coefficients, clip all negative values to 0
    and normalize to sum to 1.
    """
    coefs = coefs.copy().flatten()
    coefs[coefs < 0] = 0
    coefs /= coefs.sum()  # Normalize to sum to 1
    return coefs
