from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator


@dataclass
class MartingaleModelConfig:
    width: float = 0.2
    default_num_samples: int = 1000
    random_state: Optional[int] = None


class BaseMartingaleModel(BaseModel):
    mean: Union[List[float], np.ndarray]
    cov: Union[List[List[float]], np.ndarray]
    config: MartingaleModelConfig = MartingaleModelConfig()

    class Config:
        arbitrary_types_allowed = True

    @field_validator("mean")
    def validate_mean(cls, v) -> np.ndarray:
        """Validate and convert mean to numpy array."""
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"mean must be 1-dimensional, got shape {arr.shape}")
        return arr

    @field_validator("cov")
    def validate_cov(cls, v) -> np.ndarray:
        """Validate and convert covariance matrix to numpy array."""
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"cov must be 2-dimensional, got shape {arr.shape}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"cov must be a square matrix, got shape {arr.shape}")
        return arr

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Ensure arrays are properly converted
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)

        # Dimension compatibility check
        if len(self.mean) != self.cov.shape[0]:
            raise ValueError(
                f"Dimension mismatch: mean has {len(self.mean)} elements, "
                f"cov is {self.cov.shape[0]}x{self.cov.shape[1]}"
            )

    def _validate_num_samples(self, num_samples: int) -> None:
        """Validate num_samples parameter."""
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                f"num_samples must be a positive integer, got {num_samples}"
            )

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Method to sample from the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
