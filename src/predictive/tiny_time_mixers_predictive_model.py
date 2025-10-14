import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
    get_datasets,
)
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits

from base_predictive_model import BaseConfig, BasePredictiveModel


@dataclass
class TinyTimeMixersConfig(BaseConfig):
    exogenous_variables: bool = True
    model_revision: str = "512-96-ft-r2.1"
    output_dir: str = os.path.join("..", "models", "TinyTimeMixers")
    decoder_mode: str = "mix_channel"
    fcm_context_length: int = 5
    fcm_mix_layers: int = 2
    resolution_prefix_tuning: bool = True
    fcm_use_mixer: bool = True
    enable_forecast_channel_mixing: bool = True
    fcm_prepend_past: bool = True


class TinyTimeMixersPredictiveModel(BasePredictiveModel):
    model: Optional[TinyTimeMixerForPrediction] = None
    tsp: Optional[TimeSeriesPreprocessor] = None
    best_model_path: Optional[str] = None
    control_columns: Optional[List[str]] = None
    zero_shot_configuration: Optional[bool] = True
    config: TinyTimeMixersConfig = TinyTimeMixersConfig()

    def initialize_preprocessor(self):
        if self.config.exogenous_variables and self.control_columns is None:
            control_columns = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        elif self.control_columns is None:
            control_columns = []
        else:
            control_columns = self.control_columns

        if self.target_columns is None or self.target_columns == []:
            raise ValueError("target_columns must be provided if tsp is not given.")

        self.tsp = TimeSeriesPreprocessor(
            timestamp_column=self.timestamp_column,
            target_columns=self.target_columns,
            control_columns=control_columns,
            context_length=self.config.backcast_length,
            prediction_length=self.config.forecast_length,
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",  #  type: ignore
        )

    def _prepare_output_directory(self):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

    def _validate_initialization(self):
        if (
            self.target_columns is None or self.target_columns == []
        ) and self.tsp is None:
            raise ValueError("At least one of target_columns or tsp must be provided.")

    def _freeze_backbone(self) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized.")
        # Freeze the backbone of the model
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def load_model(self):
        if self.load_local:
            if not os.path.exists(os.path.join(self.config.output_dir, "final_model")):
                raise FileNotFoundError(
                    "The final model directory does not exist. Please train the model first."
                )
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                os.path.join(self.config.output_dir, "final_model")
            )
        else:
            if self.tsp is None:
                raise ValueError("TimeSeriesPreprocessor is not initialized.")

            if self.zero_shot_configuration:
                self.model = TinyTimeMixerForPrediction.from_pretrained(
                    "ibm-granite/granite-timeseries-ttm-r2",
                    revision=self.config.model_revision,
                    context_length=self.config.backcast_length,
                    prediction_length=self.config.forecast_length,
                    num_input_channels=self.tsp.num_input_channels,
                    prediction_channel_indices=self.tsp.prediction_channel_indices,
                    exogenous_channel_indices=self.tsp.exogenous_channel_indices,
                )
            else:
                self.model = TinyTimeMixerForPrediction.from_pretrained(
                    "ibm-granite/granite-timeseries-ttm-r2",
                    revision=self.config.model_revision,
                    context_length=self.config.backcast_length,
                    prediction_length=self.config.forecast_length,
                    num_input_channels=self.tsp.num_input_channels,
                    decoder_mode=self.config.decoder_mode,
                    prediction_channel_indices=self.tsp.prediction_channel_indices,
                    exogenous_channel_indices=self.tsp.exogenous_channel_indices,
                    fcm_context_length=self.config.fcm_context_length,
                    fcm_use_mixer=self.config.fcm_use_mixer,
                    fcm_mix_layers=self.config.fcm_mix_layers,
                    enable_forecast_channel_mixing=self.config.enable_forecast_channel_mixing,
                    fcm_prepend_past=self.config.fcm_prepend_past,
                    dropout=self.config.dropout,
                    resolution_prefix_tuning=self.config.resolution_prefix_tuning,
                )
            self._freeze_backbone()

    def model_post_init(self, __context) -> None:
        self._prepare_output_directory()
        self._validate_initialization()

        if self.tsp is None:
            self.initialize_preprocessor()

        self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.random_state is not None:
            set_seed(self.config.random_state)
        self.load_model()

    def train(self, train_dataset: Any, valid_dataset: Any) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized.")

        learning_rate, self.model = optimal_lr_finder(  # type: ignore
            self.model,
            train_dataset,
            batch_size=self.config.batch_size,
            enable_prefix_tuning=False,
        )

        if self.model is None:
            raise ValueError(
                "Model is not initialized after finding optimal learning rate."
            )

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            learning_rate=(
                self.config.learning_rate
                if self.config.learning_rate is not None
                else learning_rate
            ),
            num_train_epochs=self.config.num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            dataloader_num_workers=4,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            use_cpu=self.config.device != "cuda",
        )

        assert isinstance(self.config.early_stopping_patience, int)

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=self.config.early_stopping_threshold,
        )

        tracking_callback = TrackingCallback()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=self.config.num_epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / (self.config.batch_size)),
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[early_stopping_callback, tracking_callback],
            optimizers=(optimizer, scheduler),  # type: ignore
        )

        trainer.train()
        trainer.save_model(os.path.join(self.config.output_dir, "final_model"))

    def predict(self, data: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        if self.model is None or self.tsp is None:
            raise ValueError("Model or TimeSeriesPreprocessor is not initialized.")

        pipeline = TimeSeriesForecastingPipeline(
            self.model,
            device=self.config.device,
            feature_extractor=self.tsp,
            batch_size=self.config.batch_size,
        )
        cols = self.tsp.target_columns

        forecast = pipeline(data)
        assert isinstance(forecast, pd.DataFrame)

        if dropna:
            forecast.dropna(inplace=True)
        predictions = forecast[cols + [f"{t}_prediction" for t in cols]].map(
            lambda x: x[0]
        )

        predictions[self.timestamp_column] = pd.to_datetime(data[self.timestamp_column])
        predictions.set_index(self.timestamp_column, inplace=True)
        if dropna:
            predictions.dropna(inplace=True)

        return predictions

    def get_dataset(
        self,
        data: pd.DataFrame,
        split_config: Optional[Dict[str, List[int | float] | float]] = None,
    ) -> Tuple[Any, Any, Any]:
        if split_config is None:
            split_config = {
                "train": [0, 0.6],
                "valid": [0.6, 0.75],
                "test": [0.75, 1.0],
            }
        if self.tsp is None:
            raise ValueError("TimeSeriesPreprocessor is not initialized.")

        if self.model is None:
            raise ValueError("Model is not initialized.")

        train_dataset, valid_dataset, test_dataset = get_datasets(  # type: ignore
            self.tsp,
            data,
            split_config,
            use_frequency_token=self.model.config.resolution_prefix_tuning,
        )
        return train_dataset, valid_dataset, test_dataset

    def get_dataset_as_dataframe(
        self,
        data: pd.DataFrame,
        split_config: Optional[Dict[str, List[int | float] | float]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if split_config is None:
            split_config = {
                "train": [0, 0.6],
                "valid": [0.6, 0.75],
                "test": [0.75, 1.0],
            }
        if self.tsp is None:
            raise ValueError("TimeSeriesPreprocessor is not initialized.")

        if self.model is None:
            raise ValueError("Model is not initialized.")

        train_df, valid_df, test_df = prepare_data_splits(  # type: ignore
            data,
            context_length=self.config.backcast_length,
            split_config=split_config,
        )

        return train_df, valid_df, test_df
