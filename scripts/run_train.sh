#!/bin/bash


ckpt_dir=opmodel
gpus=0

python3 -B ./train.py \
    --gpu_ids=$gpus \
    --tr_list=../filelists/tr_list.txt \
    --cv_file=../data/datasets/cv/cv.ex \
    --ckpt_dir=$ckpt_dir \
    --logging_period=1000 \
    --lr=0.0002 \
    --time_log=./time.log \
    --unit=utt \
    --batch_size=2 \
    --buffer_size=32 \
    --max_n_epochs=20 \
