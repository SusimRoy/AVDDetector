import argparse
from torch.utils.data import DataLoader, IterableDataset, Dataset
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from deepfakeloader.loader import AVDeepfake1mPlusPlusMagnifiedVideo
from avcls import AVClassifier
from utils import LrLogger, EarlyStoppingLR
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch import LightningDataModule
import os
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="Classification model training")
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model", type=str, choices=["av-classifier", "meso4", "meso_inception4"])
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=32)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=2000)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--text_file", type=str, default=None)  # <--- Add this line
args = parser.parse_args()

class AVDeepfake1mPlusPlusMagnifiedVideoDataModule(LightningDataModule):
    def __init__(self, data_root, batch_size=2, num_workers=2, num_train=None, num_val=None, text_file=None):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.text_file = text_file

        self.num_val = num_val

    def setup(self, stage=None):
        # self.train_dataset = AVDeepfake1mPlusPlusMagnifiedVideo(
        #     subset="train",
        #     data_root=self.data_root,
        #     take_num=self.num_train,
        #     use_video_label=True
        # )
        self.test_dataset = AVDeepfake1mPlusPlusMagnifiedVideo(
            subset="val",
            data_root=self.data_root,
            take_num=self.num_train,
            use_video_label=True,
            text_file = self.text_file
        )
        # You can add val/test datasets here if needed

    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=True,  # Lightning will replace with DistributedSampler in DDP
    #         pin_memory=True,
    #     )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Lightning will replace with DistributedSampler in DDP
            pin_memory=True,
            drop_last=False
        ) 

if __name__ == "__main__":

    learning_rate = 1e-5
    gpus = args.gpus
    total_batch_size = args.batch_size * gpus
    learning_rate = learning_rate * total_batch_size / 4

    if args.model == "av-classifier":
        model = AVClassifier.load_from_checkpoint(args.resume, lr = learning_rate, distributed=True)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    valloader = AVDeepfake1mPlusPlusMagnifiedVideoDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=2,
        num_train=args.num_train,
        num_val=args.num_val,
        text_file = args.text_file    
        )

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    monitor = "train_epoch_auc"

    trainer = Trainer(
        log_every_n_steps=50,
        precision=precision,
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./ckpt/{args.model}",
                save_last=True,
                filename=args.model + "-{epoch}-{train_epoch_auc:.3f}",
                monitor=monitor,
                mode="max"
            ),
            LrLogger(),
            EarlyStoppingLR(lr_threshold=1e-7)
        ],
        enable_checkpointing=True,
        benchmark=True,
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        sync_batchnorm=True, 
    )
    trainer.test(model, valloader)