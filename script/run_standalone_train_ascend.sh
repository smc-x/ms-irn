#!/bin/bash
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1
if [ $# != 2 ]; then
    echo "Usage: \
bash run_standalone_train_ascend.sh [SCALE] [DATASET_GT_PATH]"
    exit 1
fi
if [ ! -d $2 ]; then
    echo "error: DATASET_PATH:$2 does not exist"
    exit 1
fi
nohup python ../train.py \
    --scale $1 \
    --device_target Ascend \
    --dataset_GT_path $2 \
    > ../rain.log 2>&1 &
pid=$!
echo "Start training with rank ${RANK_ID} on device ${DEVICE_ID}: ${pid}"