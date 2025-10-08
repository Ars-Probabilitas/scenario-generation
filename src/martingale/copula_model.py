from typing import Optional

import numpy as np
import pandas as pd
from base_martingale_model import BaseMartingaleModel
from copulas.multivariate import GaussianMultivariate


class CopulaModel(BaseMartingaleModel):
    """Sampling using Gaussian copulas.

    This class uses a Gaussian copula to model the dependence structure
    and generate samples accordingly."""

    copula: Optional[GaussianMultivariate] = None

    def model_post_init(self, __context) -> None:
        """Initialize and validate copula."""
        super().model_post_init(__context)
        if self.copula is None:
            raise ValueError("Copula must be provided for CopulaSampling")

    def sample(
        self,
        num_samples: int,
    ) -> np.ndarray:
        """Sample from copula distribution."""
        self._validate_num_samples(num_samples)

        if self.copula is None:
            raise ValueError("Copula must be set before sampling")

        return self.copula.sample(num_samples).to_numpy()

    def __str__(self) -> str:
        return f"CopulaSampling(dimensions={self.mu.shape[0]})"
