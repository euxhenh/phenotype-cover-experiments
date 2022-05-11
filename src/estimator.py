import abc


class BaseEstimator:
    @abc.abstractmethod
    def __init__(self):
        self.__fitted__ = False

    @abc.abstractmethod
    def fit(self, **kwargs):
        self.__fitted__ = True
