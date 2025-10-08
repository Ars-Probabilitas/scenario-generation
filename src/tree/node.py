from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator


class TreeNode(BaseModel):
    """Represents a node in a scenario tree for stochastic processes."""

    id: int
    parent_id: Optional[int]
    model_input: np.ndarray
    scenario: np.ndarray
    martingale: np.ndarray
    probability: float
    is_leaf: bool

    class Config:
        arbitrary_types_allowed = True

    @field_validator("id")
    def validate_id(cls, v):
        if v < 0:
            raise ValueError("Node ID must be positive")
        return v

    @field_validator("probability")
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("model_input", "scenario", "martingale")
    def validate_arrays(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.size == 0:
            raise ValueError("Array cannot be empty")
        return v
