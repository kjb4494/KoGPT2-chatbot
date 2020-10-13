# -*- coding: utf-8 -*-
import logging
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from kogpt2_chat import KoGPT2Chat, get_kogpt2_args

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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


def test_code():
    import argparse

    parser = argparse.ArgumentParser(description='KoGPT-2 Test Train')
    parser.add_argument('--gpus',
                        type=str,
                        default='0',
                        help='gpu count for use.')

    parser.add_argument('--max_epochs',
                        type=str,
                        default='1',
                        help='max epochs.')

    args = parser.parse_args()

    train(
        gpus=args.gpus,
        max_epochs=args.max_epochs
    )


# test_code
if __name__ == "__main__":
    test_code()
