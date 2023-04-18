"""
Original source: https://github.com/facebookresearch/pycls/blob/master/pycls/core/config.py
Latest commit 2c152a6 on May 6, 2021
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

# from .io import cache_url, pathmgr
from yacs.config import CfgNode

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Model options ----------------------------------- #
_C.MODEL = CfgNode()

# Model backbone
_C.MODEL.BACKBONE = 'resnet_1'

_C.MODEL.FREEZE_BACKBONE = False

_C.MODEL.FREEZE_BATCHNORM = True

# Sequence length
_C.MODEL.SEQ_LEN = 50

# Multi-level feature
_C.MODEL.LEVEL = 's0'  # Only last output

# Use TPP features
_C.MODEL.TPP_FEAT = True

# Number of units for prediction heads
_C.MODEL.PREDICTION_DIM = 1024

# Output logits or probabilities
_C.MODEL.FROM_LOGITS = True

# Output size of temporal pyramid pooling distance
_C.MODEL.OUTPUT_DIM = 1024

# Similarity decoder block dilation rates
_C.MODEL.SD_DILATION_RATES = -1

# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

_C.OPTIM.NAME = 'adam'

_C.OPTIM.BASE_LR = 1e-4

# LR warmup
_C.OPTIM.WARMUP_STEPS = 0.

# LR Scheduler
_C.OPTIM.LR_SCHEDULER = 'cosine'
_C.OPTIM.LR_SCHEDULER_DECAY_STEPS = -1
_C.OPTIM.LR_SCHEDULER_ALPHA = 0.01


_C.OPTIM.WEIGHT_DECAY = 5e-5

_C.OPTIM.MOMENTUM = 0.9
# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 16

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = True

# Number of epochs
_C.TRAIN.EPOCHS = 10

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

# Total mini-batch size
_C.TEST.BATCH_SIZE = 16

# Weights to use for testing
_C.TEST.WEIGHTS = ""

# ---------------------------------- Dataloader options ----------------------------------- #
_C.DATA_LOADER = CfgNode()

# ROOT of DATASET
_C.DATA_LOADER.DATA_DIR = '/mnt/XProject/dataset'

# Image size
_C.DATA_LOADER.IMG_SIZE = 224

# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = True

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "./tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "run_config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Logger (wandb or TensorBoard)
_C.LOGGER = "TensorBoard"

# Debug (run eagerly)
_C.DEBUG = False

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    # err_str = "The first lr step must start at 0"
    # assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str

    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    # err_str = "Log destination '{}' not supported"
    # assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)
    err_str = "Sequence length '{}' must divisible by 10"
    assert _C.MODEL.SEQ_LEN % 10 == 0, err_str.format(_C.MODEL.SEQ_LEN)


def dump_cfg(out_dir=''):
    """Dumps the config to the output directory."""
    if out_dir == '':
        out_dir = _C.OUT_DIR

    cfg_dir = pathlib.Path(out_dir)
    cfg_dir.mkdir(exist_ok=True, parents=True)
    cfg_file = cfg_dir / _C.CFG_DEST

    with open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "See core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    load_cfg(args.cfg_file)
    _C.merge_from_list(args.opts)
