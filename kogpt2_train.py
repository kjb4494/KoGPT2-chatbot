# -*- coding: utf-8 -*-
import logging
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from kogpt2_chat import KoGPT2Chat, get_kogpt2_args

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train(gpus, max_epochs):
    args = get_kogpt2_args(is_train=True)
    args.gpus = gpus
    args.max_epochs = max_epochs
    logging.info(args)

    checkpoint_callback = ModelCheckpoint(
        filepath='model_chp/{epoch:02d}-{loss:.2f}',
        verbose=True,
        save_last=True,
        monitor='loss',
        mode='min',
        prefix='model_'
    )
    # python train_torch.py --train --gpus 1 --max_epochs 3
    model = KoGPT2Chat(args)
    model.train()
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
    trainer.fit(model)
    logging.info('best model path {}'.format(checkpoint_callback.best_model_path))


# test_code
if __name__ == "__main__":
    train(gpus=1, max_epochs=2)
