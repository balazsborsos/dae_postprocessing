import torch
from pathlib import Path
import pytorch_lightning as pl
from typing import Union

from models.DAE import DAE
from utils.loss import DiceLoss
from utils.data import get_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def get_model(
        *args,
        **kwargs
) -> Union[torch.nn.Module, pl.LightningModule]:
    return DenoiseModel(**kwargs)


class DenoiseModel(DAE, pl.LightningModule):

    def __init__(self, *args, **kwargs):
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        del kwargs['in_channels'], kwargs['out_channels']
        super(DenoiseModel, self).__init__(spatial_dims=2,
                                           in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           **kwargs)

        self.lr = kwargs["learning_rate"]
        self.dice_loss_func = DiceLoss()

    def training_step(self, batch, batch_idx):
        x, mask = batch
        mask_out = self.forward(x)

        loss = self.dice_loss_func(mask_out, mask).mean()
        self.log(f"train_denoise_dice", loss, prog_bar=True, on_epoch=True)

        if self.current_epoch % 3 == 0:
            if batch_idx <= 20 and batch_idx % 2 == 0:
                for i in range(x.shape[0]):
                    img_ = self.create_collage(x[i][0], mask_out[i][0], mask[i][0])
                    if isinstance(self.logger.experiment, list):
                        self.logger.experiment[0].add_image(f'train_batch_{batch_idx}_{i}_mask', img_,
                                                            self.current_epoch)
                    else:
                        self.logger.experiment.add_image(f'train_batch_{batch_idx}_{i}_mask', img_, self.current_epoch)
                    del img_
        del x, mask_out

        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        mask_out = self.forward(x)

        loss = self.dice_loss_func(mask_out, mask).mean()
        self.log(f"val_denoise_dice", loss)

        if self.current_epoch % 3 == 0:
            if batch_idx <= 20 and batch_idx % 2 == 0:
                for i in range(x.shape[0]):
                    img_ = self.create_collage(x[i][0], mask_out[i][0], mask[i][0])
                    if isinstance(self.logger.experiment, list):
                        self.logger.experiment[0].add_image(f'validation_batch_{batch_idx}_{i}_mask', img_,
                                                            self.current_epoch)
                    else:
                        self.logger.experiment.add_image(f'validation_batch_{batch_idx}_{i}_mask', img_,
                                                         self.current_epoch)
                    del img_
        del x, mask_out
        del loss

    @staticmethod
    @torch.no_grad()
    def create_collage(x_i, out_i, y_i):
        # Normalize middle image in range [0, 1], make it into 3-channel BGR image
        x_norm = (x_i - x_i.min()) / (x_i.max() - x_i.min())
        x_norm = torch.stack((x_norm, x_norm, x_norm))
        mask_out_s = torch.stack((out_i, out_i, out_i))
        mask_s = torch.stack((y_i, y_i, y_i))
        del x_i

        def create_row(img, mask_out, mask):
            # For each mask, create an overlay over the image and append to current row
            row = torch.concat((img, mask_out, mask), dim=2)
            return row

        img_ = create_row(x_norm, mask_out_s, mask_s)
        del x_norm, y_i, out_i
        img_ *= 255
        img_ = img_.cpu().to(torch.uint8)
        return img_

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_model(config: dict, data_path: Path):
    model_dir = config["store_model_to"]
    print('Model directory: ', model_dir)

    # custom parameters set in yaml file
    training_params = config["training_configuration"]

    # callback and logger
    callbacks = [
        ModelCheckpoint(  # saves weights for every n epochs
            dirpath=Path(model_dir, "checkpoint"),
            filename='weights.epoch{epoch:03}-val_denoise_dice_{val_denoise_dice:.4f}',
            save_top_k=-1,
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=5,
        )
    ]

    loggers = [
        TensorBoardLogger(
            save_dir=model_dir,
            name='board',
            version=''
        ),
    ]
    # model
    model = get_model(**training_params)

    # data
    train_dataloader = get_dataloader(
        data_path,
        mlset="training",
        **training_params
    )
    val_dataloader = get_dataloader(
        data_path,
        mlset="validation",
        **training_params
    )

    # training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=training_params["max_epochs"],
        gpus=training_params.get("gpus", 1),
        auto_select_gpus=True,
        strategy=training_params.get("strategy", None),
        gradient_clip_val=0.5,
        log_every_n_steps=5
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
