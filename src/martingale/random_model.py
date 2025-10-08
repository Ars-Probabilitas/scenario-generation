from typing import Optional

import numpy as np
import pandas as pd
from base_martingale_model import BaseMartingaleModel


class RandomModel(BaseMartingaleModel):
    """Standard multivariate normal sampling using numpy's random generator.

    This class implements direct sampling from a multivariate normal distribution
    using the mean vector and covariance matrix.
    """

    def sample(
        self,
        num_samples: int,
    ) -> np.ndarray:
        """
        Sample from a multivariate normal distribution.
        """
        self._validate_num_samples(num_samples)

        # Set random seed if provided
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)

        return np.random.multivariate_normal(self.mu, self.cov, num_samples)

    def __str__(self):
        return f"RandomSampling(mu={self.mu}, cov={self.cov})"
