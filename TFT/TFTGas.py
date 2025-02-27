
import pandas as pd 
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import MAPE, RMSE
import torch.nn as nn
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
import matplotlib.pyplot as plt
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from memory_profiler import profile

def main():
    data = pd.read_csv(r"TFTDatasets\TFTGas.csv", index_col=0, sep=',', decimal='.', encoding = "ISO-8859-1")
    max_prediction_length = 1
    max_encoder_length = 100
    training_cutoff = data["days_from_start"].max() - max_prediction_length
    print("instanciating training function is being ran")
    training = TimeSeriesDataSet(
        data[data["days_from_start"] <= training_cutoff],
        time_idx="days_from_start",
        target="gas_use",
        group_ids=["id"],
        min_encoder_length = max_encoder_length // 2, 
        max_encoder_length = max_encoder_length,
        min_prediction_length = 1,
        max_prediction_length = max_prediction_length,
        static_categoricals=["id", "income_band", "hometype", "build_era"], #social variables
        static_reals=["residents"], #social variables
        time_varying_known_reals=["temperature", "apparent_temperature", "precipitation", "wind_speed"], #weather variables
        time_varying_known_categoricals=["day","day_of_week", "month", "year"], #calendar variables
        time_varying_unknown_reals=["gas_use"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, stop_randomization=True, predict=True)
    batch_size = 32
 
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=2)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(           
        max_epochs=40,
        accelerator="cpu",
        devices="1",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        logger=logger,
        callbacks=[lr_logger, early_stop_callback],
        limit_test_batches=50,
        )

    tft = TemporalFusionTransformer.from_dataset(         
        training,
        learning_rate = 0.01,
        hidden_size = 32,
        attention_head_size=8,
        dropout = 0.1,
        hidden_continuous_size=4,
        output_size=1, 
        loss=MAPE(),
        log_interval=10,
        optimizer="Ranger",
        reduce_on_plateau_patience=3,
        lstm_layers=2
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
  
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("Found the path" + best_model_path + "<---- Here")
    
    #Change path below to a .ckpt file to test performence of a saved model
    # best_model_path = r"C:\Users\sukhy\Documents\Thesis\Models\TFT\Gas\Calendar\Calandar_1\checkpoints\epoch=6-step=1792.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)



    prediction = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    print(prediction.y)
    raw = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    mape = MAPE()
    rmse = RMSE()
    print(mape(prediction.output, prediction.y))
    print(rmse(prediction.output, prediction.y))
if __name__ == '__main__':
    main()