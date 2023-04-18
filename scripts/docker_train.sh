#!/bin/bash

trap "exit" INT

sleep 1
docker run --gpus all --ipc=host -u $(id -u):$(id -g) -it --rm \
  -v /home/hvthong/sXProject/GEBD:/home/hvthong/sXProject/GEBD \
  -v /mnt/SharedProject/Dataset/LOVEU_22/gebd:/mnt/SharedProject/Dataset/LOVEU_22/gebd \
  -v /mnt/Work/Dataset:/mnt/Work/Dataset \
  -w /home/hvthong/sXProject/GEBD \
  gebd:tf2.9.1 bash scripts/train.sh
