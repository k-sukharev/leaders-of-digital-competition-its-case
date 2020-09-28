import os
import sys

import imageio
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

from argparse import ArgumentParser


from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_toolbelt import losses as L
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader

from utils.dataset import BacteriaDataset, LABEL_TO_IDX
from utils.helpers import get_img_idxs, fix_seed, process_json
from utils.transforms import train_transform, valid_transform


class SegmentationModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        net = getattr(smp, self.hparams.architecture)

        self.net = net(
            encoder_name=self.hparams.encoder_name,
            classes=1
        )

        self.iou = smp.utils.metrics.IoU(activation='sigmoid')

        self.loss = L.JointLoss(
            L.BinaryFocalLoss(),
            L.BinaryLovaszLoss(),
            0.7,
            0.3
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']

        mask_logits = self.forward(image)

        loss = self.loss(mask_logits, mask_gt)

        iou = self.iou(mask_logits, mask_gt)

        log_dict = {
            'train_loss': loss,
            'iou': iou
        }

        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        mask_gt = batch['mask']

        mask_logits = self.forward(image)

        loss = self.loss(mask_logits, mask_gt)

        val_iou = self.iou(mask_logits, mask_gt)

        return {
            'val_loss': loss,
            'val_iou': val_iou
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_iou = torch.stack([x['val_iou'] for x in outputs]).mean()

        log_dict = {
            'val_loss': val_loss,
            'val_iou': val_iou
        }

        return {
            'val_loss': val_loss,
            'log': log_dict,
            'progress_bar': log_dict
        }

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {
                'params': self.net.encoder.parameters(),
                'lr': self.hparams.encoder_lr,
                'weight_decay': self.hparams.encoder_weight_decay
            },
            {'params': self.net.decoder.parameters()},
            {'params': self.net.segmentation_head.parameters()},
        ], lr=self.hparams.lr)

        scheduler = {
            'scheduler': optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.scheduler_gamma
            ),
            'interval': 'epoch',
            'frequency': self.hparams.scheduler_frequency,
            'reduce_on_plateau': False,
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]

    def prepare_data(self):
        path = os.path.join(self.hparams.data_path, 'train')
        img_idxs = get_img_idxs(path)

        images = []
        masks = []
        labels = []

        for img_idx in img_idxs:
            image_path = os.path.join(path, f'{img_idx}.png')
            image = imageio.imread(image_path)
            images.append(image)

            json_path = os.path.join(path, f'{img_idx}.json')
            mask, label = process_json(json_path)
            masks.append(mask)
            labels.append(label)

        images = np.stack(images)
        labels = np.array(list(map(lambda x: LABEL_TO_IDX[x], labels)))
        masks = np.stack(masks)

        skf = StratifiedKFold(
            n_splits=self.hparams.n_splits,
            shuffle=True,
            random_state=self.hparams.seed
        )

        folds = list(skf.split(np.zeros(len(labels)), labels))

        train_images, valid_images = (
            images[folds[self.hparams.fold][0]],
            images[folds[self.hparams.fold][1]]
        )
        train_labels, valid_labels = (
            labels[folds[self.hparams.fold][0]],
            labels[folds[self.hparams.fold][1]]
        )
        train_masks, valid_masks = (
            masks[folds[self.hparams.fold][0]],
            masks[folds[self.hparams.fold][1]]
        )

        self.train_dataset = BacteriaDataset(
            train_images,
            labels=train_labels,
            masks=train_masks,
            transform=train_transform,
        )

        self.valid_dataset = BacteriaDataset(
            valid_images,
            labels=valid_labels,
            masks=valid_masks,
            transform=valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )


def main(hparams):
    if torch.cuda.is_available():
        torch.cuda.set_device(hparams.gpu)

    fix_seed(hparams.seed)

    model = SegmentationModel(hparams)

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_iou',
        mode='max'
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=hparams.accumulate_batches,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=1,
        gpus=[hparams.gpu] if torch.cuda.is_available() else None,
        max_epochs=hparams.epochs,
        deterministic=True
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='select GPU device')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/',
        help='path where dataset is stored'
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--n_splits',
        type=int,
        default=10,
        help='number of folds'
    )
    parser.add_argument('--fold', type=int, default=0, help='fold index')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='number of workers for dataloader'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='size of the batches'
    )
    parser.add_argument(
        '--accumulate_batches',
        type=int,
        default=2,
        help='batches to accumulate grads'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='Unet',
        help='model architecture'
    )
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='se_resnext50_32x4d',
        help='model encoder'
    )
    parser.add_argument(
        '--encoder_lr',
        type=float,
        default=5e-4,
        help='encoder learning rate'
    )
    parser.add_argument(
        '--encoder_weight_decay',
        type=float,
        default=3e-5,
        help='encoder weight decay'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate'
    )
    parser.add_argument(
        '--scheduler_gamma',
        type=float,
        default=0.95,
        help='gamma for ExponentialLR scheduler'
    )
    parser.add_argument(
        '--scheduler_frequency',
        type=int,
        default=4,
        help='scheduler step frequency'
    )

    hparams = parser.parse_args()

    main(hparams)
