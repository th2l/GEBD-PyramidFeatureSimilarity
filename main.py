"""
Author: Van-Thong Huynh
Affiliation: Chonnam Nat'l Univ.
"""
import pathlib

from core import config
from core.config import cfg
from core.utils import set_seed, set_gpu_growth, set_mixed_precision
from core.train_lib import run_experiment

import os.path as osp
import shutil
import glob


def copyfiles(source_dir, dest_dir, ext='*.py'):
    # Copy source files or compress to zip
    files = glob.iglob(osp.join(source_dir, ext))
    for file in files:
        if osp.isfile(file):
            shutil.copy2(file, dest_dir)

    if osp.isdir(osp.join(source_dir, 'core')) and not osp.isdir(osp.join(dest_dir, 'core')):
        shutil.copytree(osp.join(source_dir, 'core'), osp.join(dest_dir, 'core'), copy_function=shutil.copy2)


if __name__ == '__main__':
    config.load_cfg_fom_args("GEBD - ACCV 2022")
    config.assert_and_infer_cfg()
    cfg.freeze()

    cfg_file = config.dump_cfg(cfg.OUT_DIR)

    # Copy source code
    out_dir_src = pathlib.Path(cfg.OUT_DIR) / 'src'
    out_dir_src.mkdir(exist_ok=True, parents=True)
    copyfiles(source_dir='./', dest_dir=out_dir_src.__str__())

    # Set seed and GPU
    set_seed(cfg.RNG_SEED)
    n_gpus = set_gpu_growth()
    set_mixed_precision(cfg.TRAIN.MIXED_PRECISION)

    # Run experiments
    run_experiment(cfg, n_gpus)

    print('Stop in here')
