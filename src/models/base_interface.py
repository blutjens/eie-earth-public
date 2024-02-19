""" A model interface that defines the API to able to use models w/ MLFlow.

TODO: complete base class.
"""

from abc import ABC, abstractmethod

import torch


class MLProjectBase(ABC):
    """ An abstract base class (ABC) that enables the use of external models
    with our MLFlow experiment logic.

    """

    @abstractmethod
    def fit():
        """
        """
        pass

    @abstractmethod
    def predict():
        """
        """
        pass
