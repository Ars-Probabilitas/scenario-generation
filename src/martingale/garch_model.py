from typing import Optional, Union

import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelResult
from base_martingale_model import BaseMartingaleModel


class GarchModel(BaseMartingaleModel):
    garch: ARCHModelResult
    scale: Optional[float] = None
    initial_sampling_date: Optional[pd.Timestamp] = None
    freq: Union[str, pd.DateOffset] = "H"

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Method to sample from the distribution.
        """
        if self.garch is None:
            raise ValueError(
                "GARCH model is not fitted. Please fit the model before sampling."
            )

        samples = self.garch.forecast(
            start=str(self.initial_sampling_date),
            method="simulation",
            horizon=1,
            simulations=num_samples,
        )

        if (
            samples is None
            or samples.simulations is None
            or samples.simulations.values is None
        ):
            raise ValueError("Sampling failed. No samples were generated.")

        self.initial_sampling_date = self._increment_timestamp()

        martingale_simulations = pd.DataFrame(samples.simulations.values[:, :, 0])
        if self.scale is not None:
            martingale_simulations /= self.scale

        return martingale_simulations.iloc[0].to_numpy()

    def _increment_timestamp(self) -> pd.Timestamp:
        if self.initial_sampling_date is None:
            raise ValueError(
                "initial_sampling_date is not set. Cannot increment timestamp."
            )
        if isinstance(self.freq, str):
            offset = pd.tseries.frequencies.to_offset(self.freq)
        else:
            offset = self.freq

        return self.initial_sampling_date + (offset)
