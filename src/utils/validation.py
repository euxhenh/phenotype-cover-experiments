def check_is_fitted(estimator, attributes=None):
    """
    Check if estimator has been fit by checking if all ``attributes``
    are present in estimator.
    """
    if not hasattr(estimator, 'fit'):
        return TypeError(f"{estimator} is not an estimator.")

    fitted = True

    if attributes is not None:
        for attr in attributes:
            if not hasattr(estimator, attr):
                fitted = False
                break
    else:
        if not hasattr(estimator, '__fitted__'):
            raise TypeError(f"{estimator} does not have a __fitted__ attr.")
        fitted = estimator.__fitted__

    if not fitted:
        raise Exception(
            f"{type(estimator).__name__} has not been fitted yet.")


def _validate_n_neighbors(n_neighbors):
    if not isinstance(n_neighbors, int):
        raise ValueError("Number of neighbors must be integer.")
    if n_neighbors < 1:
        raise ValueError("Number of neighbors must be positive.")
    return n_neighbors


def _validate_algorithm(algorithm):
    if not isinstance(algorithm, str):
        raise ValueError("Algorithm must be a string.")
    if algorithm not in ['exact', 'approx', 'auto']:
        raise ValueError("Algorithm must be one of: exact, approx, or auto.")
    return algorithm


def _validate_bool(bl):
    if not isinstance(bl, bool):
        return ValueError("Non boolean value passed.")
    return bl
