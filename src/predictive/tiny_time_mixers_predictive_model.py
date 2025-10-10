import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import torch
from base_predictive_model import BaseConfig, BasePredictiveModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
)
from tsfm_public.toolkit.lr_finder import optimal_lr_finder


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
    ttm_config: TinyTimeMixersConfig = TinyTimeMixersConfig()

    def initialize_preprocessor(self):
        if self.ttm_config.exogenous_variables and self.control_columns is None:
            control_columns = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        elif self.control_columns is None:
            control_columns = []
        else:
            control_columns = self.control_columns

        if self.target_columns is None:
            raise ValueError("target_columns must be provided if tsp is not given.")

        self.tsp = TimeSeriesPreprocessor(
            timestamp_column=self.timestamp_column,
            target_columns=self.target_columns,
            control_columns=control_columns,
            context_length=self.ttm_config.backcast_length,
            prediction_length=self.ttm_config.forecast_length,
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",  #  type: ignore
        )

    def _prepare_output_directory(self):
        if not os.path.exists(self.ttm_config.output_dir):
            os.makedirs(self.ttm_config.output_dir)

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
            if not os.path.exists(
                os.path.join(self.ttm_config.output_dir, "final_model")
            ):
                raise FileNotFoundError(
                    "The final model directory does not exist. Please train the model first."
                )
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                os.path.join(self.ttm_config.output_dir, "final_model")
            )
        else:
            if self.tsp is None:
                raise ValueError("TimeSeriesPreprocessor is not initialized.")

            self.model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r2",
                revision=self.ttm_config.model_revision,
                context_length=self.ttm_config.backcast_length,
                prediction_length=self.ttm_config.forecast_length,
                num_input_channels=self.tsp.num_input_channels,
                decoder_mode=self.ttm_config.decoder_mode,
                prediction_channel_indices=self.tsp.prediction_channel_indices,
                exogenous_channel_indices=self.tsp.exogenous_channel_indices,
                fcm_context_length=self.ttm_config.fcm_context_length,
                fcm_use_mixer=self.ttm_config.fcm_use_mixer,
                fcm_mix_layers=self.ttm_config.fcm_mix_layers,
                enable_forecast_channel_mixing=self.ttm_config.enable_forecast_channel_mixing,
                fcm_prepend_past=self.ttm_config.fcm_prepend_past,
                dropout=self.ttm_config.dropout,
                resolution_prefix_tuning=self.ttm_config.resolution_prefix_tuning,
            )
            self._freeze_backbone()

    def model_post_init(self, __context) -> None:
        self._prepare_output_directory()
        self._validate_initialization()

        if self.tsp is None:
            self.initialize_preprocessor()

        self.ttm_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.ttm_config.random_state is not None:
            set_seed(self.ttm_config.random_state)
        self.load_model()

    def train(self, train_dataset: Any, valid_dataset: Any) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized.")

        learning_rate, self.model = optimal_lr_finder(  # type: ignore
            self.model,
            train_dataset,
            batch_size=self.ttm_config.batch_size,
            enable_prefix_tuning=False,
        )

        if self.model is None:
            raise ValueError(
                "Model is not initialized after finding optimal learning rate."
            )

        args = TrainingArguments(
            output_dir=self.ttm_config.output_dir,
            overwrite_output_dir=True,
            learning_rate=(
                self.ttm_config.learning_rate
                if self.ttm_config.learning_rate is not None
                else learning_rate
            ),
            num_train_epochs=self.ttm_config.num_epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=self.ttm_config.batch_size,
            per_device_eval_batch_size=self.ttm_config.batch_size,
            dataloader_num_workers=4,
            report_to=None,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=1,
            logging_dir=os.path.join(self.ttm_config.output_dir, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            use_cpu=self.ttm_config.device != "cuda",
        )

        assert isinstance(self.ttm_config.early_stopping_patience, int)

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.ttm_config.early_stopping_patience,
            early_stopping_threshold=self.ttm_config.early_stopping_threshold,
        )

        tracking_callback = TrackingCallback()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            learning_rate,
            epochs=self.ttm_config.num_epochs,
            steps_per_epoch=math.ceil(
                len(train_dataset) / (self.ttm_config.batch_size)
            ),
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
        trainer.save_model(os.path.join(self.ttm_config.output_dir, "final_model"))

    def predict(self, data: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        if self.model is None or self.tsp is None:
            raise ValueError("Model or TimeSeriesPreprocessor is not initialized.")

        pipeline = TimeSeriesForecastingPipeline(
            self.model,
            device=self.ttm_config.device,
            feature_extractor=self.tsp,
            batch_size=self.ttm_config.batch_size,
        )
        cols = self.tsp.target_columns

        finetune_forecast = pipeline(data)
        assert isinstance(finetune_forecast, pd.DataFrame)

        if dropna:
            finetune_forecast.dropna(inplace=True)
        predictions = finetune_forecast[cols + [f"{t}_prediction" for t in cols]].map(
            lambda x: x[0]
        )

        predictions[self.timestamp_column] = pd.to_datetime(data[self.timestamp_column])
        predictions.set_index(self.timestamp_column, inplace=True)
        if dropna:
            predictions.dropna(inplace=True)

        return predictions
