from enum import Enum
from typing import Union

import numpy as np
from pydantic import BaseModel


class SolverType(Enum):
    CLIP = "clip"
    SCALE = "scale"
    NONE = "none"


class BaseConstraint(BaseModel):
    solver_type: SolverType = SolverType.NONE

    def evaluate(self, value: Union[np.ndarray, float]) -> Union[bool, np.bool_]:
        return False

    def solve(self, forecast: float, surprise: float) -> float:
        """Solve the constraint based on the forecast and surprise."""
        return 0.0

    def __str__(self):
        return f"Constraint(solver_type={self.solver_type})"


class LowerBoundConstraint(BaseConstraint):
    lower_bound: float

    def evaluate(self, value: Union[np.ndarray, float]) -> Union[bool, np.bool]:
        if isinstance(value, np.ndarray):
            return np.all(value >= self.lower_bound)
        return value >= self.lower_bound

    def solve(self, forecast: float, surprise: float) -> float:
        if self.solver_type == SolverType.CLIP:
            return self._clip(forecast + surprise)
        elif self.solver_type == SolverType.SCALE:
            return self._scale(forecast, surprise)
        else:
            raise ValueError("Type must be either 'clip' or 'scale'")

    def _clip(self, value: float) -> float:
        return max(value, self.lower_bound)

    def _scale(self, forecast: float, surprise: float) -> float:
        raise NotImplementedError(
            "Scaling not implemented for lower bound constraints."
        )

    def __str__(self):
        return f"LowerBoundConstraint(lower_bound={self.lower_bound}, solver_type={self.solver_type})"


class UpperBoundConstraint(BaseConstraint):
    upper_bound: float

    def evaluate(self, value: Union[np.ndarray, float]) -> Union[bool, np.bool]:
        if isinstance(value, np.ndarray):
            return np.all(value <= self.upper_bound)
        return value <= self.upper_bound

    def solve(self, forecast: float, surprise: float) -> float:
        if self.solver_type == SolverType.CLIP:
            return self._clip(forecast + surprise)
        elif self.solver_type == SolverType.SCALE:
            return self._scale(forecast, surprise)
        else:
            raise ValueError("Type must be either 'clip' or 'scale'")

    def _clip(self, value: float) -> float:
        return min(value, self.upper_bound)

    def _scale(self, forecast: float, surprise: float) -> float:
        raise NotImplementedError(
            "Scaling not implemented for upper bound constraints."
        )

    def __str__(self):
        return f"UpperBoundConstraint(upper_bound={self.upper_bound}, solver_type={self.solver_type})"


class IntervalConstraint(BaseConstraint):
    lower_bound: float
    upper_bound: float

    def evaluate(self, value: Union[np.ndarray, float]) -> Union[bool, np.bool]:
        if isinstance(value, np.ndarray):
            return np.all((value >= self.lower_bound) & (value <= self.upper_bound))
        return self.lower_bound <= value <= self.upper_bound

    def solve(self, forecast: float, surprise: float) -> float:
        if self.solver_type == SolverType.CLIP:
            return self._clip(forecast + surprise)
        elif self.solver_type == SolverType.SCALE:
            return self._scale(forecast, surprise)
        else:
            raise ValueError("Type must be either 'clip' or 'scale'")

    def _clip(self, value: float) -> float:
        return min(max(value, self.lower_bound), self.upper_bound)

    # TODO: Implement scaling logic for interval constraints if needed
    def _scale(self, forecast: float, surprise: float) -> float:
        if not (self.lower_bound <= forecast <= self.upper_bound):
            if self.lower_bound <= forecast + surprise <= self.upper_bound:
                return forecast + surprise
            else:
                forecast = self._clip(forecast)

        scaled_value = forecast + surprise

        if scaled_value < self.lower_bound:
            available_space = forecast - self.lower_bound
            scaled_value = forecast + available_space * surprise
        elif scaled_value > self.upper_bound:
            available_space = self.upper_bound - forecast
            scaled_value = forecast + available_space * surprise

        return scaled_value

    def __str__(self):
        return f"RangeConstraint(lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, solver_type={self.solver_type})"
