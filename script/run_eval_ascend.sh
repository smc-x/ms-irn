#!/bin/bash
export DEVICE_ID=0
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ $# != 3 ]; then
    echo "Usage: \
bash run_eval_ascend.sh [SCALE] [DATASET_PATH] [CHECKPOINT_PATH]"
    exit 1
fi

if [ ! -f $3 ]; then
    echo "error: CHECKPOINT_PATH:$3 does not exist"
    exit 1
fi

if [ ! -d $2 ]; then
    echo "error: DATASET_PATH:$2 does not exist"
    exit 1
fi

nohup python ../eval.py \
    --scale $1 \
    --dataset_GT_path $2 \
    --device_target Ascend \
    --resume_state $3 \
    > ../eval.log 2>&1 &
pid=$!
echo "Start evaluating with rank ${RANK_ID} on device ${DEVICE_ID}: ${pid}"
