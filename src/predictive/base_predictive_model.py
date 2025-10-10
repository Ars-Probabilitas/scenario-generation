from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel


@dataclass
class BaseConfig:
    backcast_length: int = 512
    forecast_length: int = 96
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: Optional[float] = None
    dropout: Optional[float] = 0.7
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None
    random_state: Optional[int] = 42
    device: str = "cpu"


class BasePredictiveModel(BaseModel):
    target_columns: Optional[List[str]] = None
    timestamp_column: str = "Datetime"
    load_local: bool = False
    config: BaseConfig = BaseConfig()

    class Config:
        arbitrary_types_allowed = True

    def train(self, train_dataset: pd.DataFrame, valid_dataset: pd.DataFrame) -> None:
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, data: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")

    def load_model(self) -> None:
        raise NotImplementedError("Subclasses should implement this method.")
