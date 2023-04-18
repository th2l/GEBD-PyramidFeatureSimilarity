"""
Author: Van Thong Huynh
Affiliation: Dept. of AI Convergence, Chonnam Nat'l Univ.
"""
import atexit
import math
import pathlib

import numpy as np
from official.modeling.optimization import CosineDecayWithOffset, LinearWarmup
from scipy.signal import argrelmax

from core.model import ProbGebd, LossScaleOptimizerV31
from core.dataloader import get_dataset

import tensorflow as tf

from tensorflow import distribute as tfdist

import keras.callbacks as kr_callbacks
from core.utils import challenge_eval_func, get_boundaries, EvalCallback, LrLogger

import wandb
from wandb.keras import WandbCallback


def get_dataloader(cfg, n_gpus=1, dataset_name='gebd'):
    data_path = cfg.DATA_LOADER.DATA_DIR
    target_fps = int(cfg.MODEL.SEQ_LEN / 10)
    target_size = cfg.DATA_LOADER.IMG_SIZE
    bsz = {'train': cfg.TRAIN.BATCH_SIZE, 'val': cfg.TEST.BATCH_SIZE}
    dataset_loader = dict()

    if dataset_name == 'TAPOS':
        splits = ['train', ]
    else:
        splits = ['train', 'val']

    for split in splits:
        dataset_loader[split] = get_dataset(data_path, target_fps=target_fps, target_size=target_size, split=split,
                                            batch_size=bsz[split], n_gpus=n_gpus)

    return dataset_loader, target_fps


def get_model(cfg, dataset_name='gebd'):
    seq_len = cfg.MODEL.SEQ_LEN
    backbone = cfg.MODEL.BACKBONE
    prediction_dim = cfg.MODEL.PREDICTION_DIM
    level = cfg.MODEL.LEVEL
    tpp_feat = cfg.MODEL.TPP_FEAT
    from_logits = cfg.MODEL.FROM_LOGITS
    output_dim = cfg.MODEL.OUTPUT_DIM
    sd_dilation_rates = cfg.MODEL.SD_DILATION_RATES

    # Create model
    model = ProbGebd(seq_len=seq_len, from_logits=from_logits, level=level, prediction_dim=prediction_dim,
                     tpp_feat=tpp_feat, output_dim=output_dim, sd_dilation_rates=sd_dilation_rates)

    # Setting optimizer
    if dataset_name == 'TAPOS':
        steps_per_epoch = int(20900 / cfg.TRAIN.BATCH_SIZE)
    else:
        steps_per_epoch = int(18800 / cfg.TRAIN.BATCH_SIZE)
    base_lr = cfg.OPTIM.BASE_LR

    n_warmup_steps = int(cfg.OPTIM.WARMUP_STEPS * steps_per_epoch)

    if cfg.OPTIM.LR_SCHEDULER_DECAY_STEPS <= cfg.OPTIM.WARMUP_STEPS:
        decay_steps = int((cfg.TRAIN.EPOCHS - cfg.OPTIM.WARMUP_STEPS) * steps_per_epoch)
    else:
        decay_steps = int(cfg.OPTIM.LR_SCHEDULER_DECAY_STEPS * steps_per_epoch)

    print('lr warmup steps: {}   Cosine decay steps: {}'.format(n_warmup_steps, decay_steps))

    if cfg.OPTIM.LR_SCHEDULER == 'cosine':
        lr_scheduler = CosineDecayWithOffset(offset=n_warmup_steps, initial_learning_rate=base_lr,
                                             decay_steps=decay_steps, alpha=cfg.OPTIM.LR_SCHEDULER_ALPHA)
    else:
        lr_scheduler = base_lr
    if n_warmup_steps > 0:
        lr_opt = LinearWarmup(after_warmup_lr_sched=lr_scheduler, warmup_steps=n_warmup_steps,
                              warmup_learning_rate=0.)
    else:
        lr_opt = base_lr if lr_scheduler is None else lr_scheduler

    # lr_opt = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=base_lr,
    #                                                            first_decay_steps=n_warmup_steps, t_mul=1.5, m_mul=1.,
    #                                                            alpha=cfg.OPTIM.LR_SCHEDULER_ALPHA)

    wd = cfg.OPTIM.WEIGHT_DECAY
    if cfg.OPTIM.NAME == 'adam':
        opt = tf.keras.optimizers.experimental.Adam(learning_rate=lr_opt, use_ema=False)
    elif cfg.OPTIM.NAME == 'adamw':
        opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_opt, weight_decay=wd, use_ema=False)
    else:
        momentum = cfg.OPTIM.MOMENTUM
        opt = tf.keras.optimizers.experimental.SGD(learning_rate=lr_opt, momentum=momentum, use_ema=False)

    if cfg.TRAIN.MIXED_PRECISION:
        # opt = keras.mixed_precision.LossScaleOptimizer(opt)
        opt = LossScaleOptimizerV31(opt)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    # model.load_weights('/mnt/Work/Dataset/Gebd-ACCV22/train_logs/20220715_170504/ckpt')

    model.compile(optimizer=opt, loss=loss, run_eagerly=cfg.DEBUG)  # loss=loss,
    return model


def run_experiment(cfg, n_gpus=1):
    if cfg.DATA_LOADER.DATA_DIR.endswith('gebd'):
        dataset_name = 'Kinetics-GEBD'
    elif cfg.DATA_LOADER.DATA_DIR.endswith('tapos'):
        dataset_name = 'TAPOS'
    else:
        raise ValueError('Unknown dataset name with data_dir '.format(cfg.DATA_LOADER.DATA_DIR))

    if not cfg.DEBUG and n_gpus == 1 and cfg.LOGGER == 'wandb':
        wandb.init(project=dataset_name + '_v2', config=cfg, save_code=True, id=cfg.OUT_DIR.split('/')[-1],
                   dir=pathlib.Path(cfg.OUT_DIR).parent.__str__())
        wandb.run.log_code()

    n_epoch = cfg.TRAIN.EPOCHS

    tsb_logdir = pathlib.Path(cfg.OUT_DIR) / 'logs'
    tsb_logdir.mkdir(exist_ok=True)

    dataloader, cur_fps = get_dataloader(cfg, n_gpus=n_gpus, dataset_name=dataset_name)

    if n_gpus > 1:
        strategy = tfdist.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = get_model(cfg, dataset_name=dataset_name)
    else:
        model = get_model(cfg, dataset_name=dataset_name)

    log_freq = 500
    callbacks = [kr_callbacks.TerminateOnNaN(),
                 # LrLogger(log_freq=log_freq, log_level='step'),
                 kr_callbacks.TensorBoard(log_dir=tsb_logdir.__str__(), update_freq=log_freq),
                 # EvalCallback(dataloader['val'], cur_fps=cur_fps)
                 ]

    if not cfg.DEBUG and cfg.LOGGER == 'wandb':
        callbacks.append(WandbCallback(save_model=False, monitor='loss', log_batch_frequency=log_freq))

    try:
        model.fit(dataloader['train'], epochs=n_epoch, callbacks=callbacks)
    except KeyboardInterrupt:
        print('Keyboard Interrupt')

    ckpt_path = pathlib.Path(cfg.OUT_DIR) / 'ckpt_last.h5'
    # ckpt_path.mkdir(exist_ok=True)
    model.save_weights(ckpt_path)

    # model.load_weights('/mnt/Work/Dataset/GEBD/train_logs/20220916_004249/ckpt_last.h5')

    if dataset_name == 'TAPOS':
        ##### TAPOS evaluation #####
        from core.tapos_utils import tapos_evaluation
        scores = tapos_evaluation(model, split='val')

        ##### End of TAPOS evaluation #####
    else:
        y_pred, y_vid_id, y_vid_dur = model.predict(dataloader['val'])

        np.save((tsb_logdir.parent / 'pred.npy').__str__(), y_pred)
        np.save((tsb_logdir.parent / 'pred_id.npy').__str__(), y_vid_id)
        np.save((tsb_logdir.parent / 'pred_dur.npy').__str__(), y_vid_dur)

        pred_bnd = get_boundaries(y_pred, y_vid_id, y_vid_dur, cur_fps=cur_fps)

        scores = challenge_eval_func(pred_dict=pred_bnd, verbose=True)

    print(scores, cur_fps)
    write_str = '{},{:.4f},{:.4f},{:.4f}\n'.format(cfg.OUT_DIR, scores[0], scores[1], scores[2])
    print(write_str)
    with open('logs.csv', 'a') as fd:
        fd.write(write_str)

    if not cfg.DEBUG and n_gpus == 1 and cfg.LOGGER == 'wandb':
        wandb.log({'val-precision': scores[0], 'val-recall': scores[1], 'val-f1': scores[2]}, commit=True)
        wandb.finish()

    if n_gpus > 1:
        atexit.register(strategy._extended._collective_ops._pool.close)
