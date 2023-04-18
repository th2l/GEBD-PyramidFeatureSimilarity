#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068
export WANDB_API_KEY='YOUR WANDB API KEY'
export WANDB_CONFIG_DIR='/mnt/Work/Dataset/GEBD/'
export WANDB_CACHE_DIR='/mnt/Work/Dataset/GEBD/'
export KERAS_HOME='/mnt/Work/Dataset/GEBD/.keras/'

data_dir='/mnt/SharedProject/Dataset/LOVEU_22/gebd'
#data_dir='/mnt/SharedProject/Dataset/LOVEU_22/tapos'

batch_size=8
# Learning rate
lr_scheduler='cosine'
lr=4e-4
alpha=0.01
n_epochs=10
n_warmup=2.
decay_steps=8

from_logits=False

seq_len=50

# Optimizer
wd=0.004
optim_func='adam'

# Network params
output_dim=256
prediction_dim=1024

multi_level=3
tpp_feat=True
sd_dlr=-1
# multi_level= 3 => /mnt/Work/Dataset/GEBD/train_logs/20220829_234823,0.7150,0.8253,0.7662
# multi_level= 3 => /mnt/Work/Dataset/GEBD/train_logs/20220904_041219,0.7076,0.8450,0.7702, batch_size = 8

for multi_level in "s1230" "s20"; do
  log_dir='/mnt/Work/Dataset/GEBD/train_logs/'$(date '+%Y%m%d_%H%M%S')
  python main.py --cfg config/base_config.yaml \
    MODEL.SEQ_LEN $seq_len \
    MODEL.LEVEL $multi_level \
    MODEL.TPP_FEAT $tpp_feat \
    MODEL.PREDICTION_DIM $prediction_dim \
    MODEL.OUTPUT_DIM $output_dim \
    MODEL.SD_DILATION_RATES $sd_dlr \
    MODEL.FROM_LOGITS $from_logits \
    OPTIM.NAME $optim_func \
    OPTIM.BASE_LR $lr \
    OPTIM.WEIGHT_DECAY $wd \
    OPTIM.LR_SCHEDULER_ALPHA $alpha \
    OPTIM.LR_SCHEDULER $lr_scheduler \
    OPTIM.WARMUP_STEPS $n_warmup \
    OPTIM.LR_SCHEDULER_DECAY_STEPS $decay_steps \
    TRAIN.MIXED_PRECISION True \
    TRAIN.BATCH_SIZE $batch_size \
    TRAIN.EPOCHS $n_epochs \
    TEST.BATCH_SIZE $batch_size \
    DATA_LOADER.DATA_DIR $data_dir \
    OUT_DIR $log_dir \
    LOGGER 'wandb'
  break
done
