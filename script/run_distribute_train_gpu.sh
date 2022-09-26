#!/bin/bash
if  [ $# != 3 ]
then
    echo "Usage:\
          bash run_distribute_train_gpu.sh [DEVICE_NUM] [SCALE] [DATASET_PATH]
          "
exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEVICE_NUM=$1
export SCALE=$2
export HCCL_CONNECT_TIMEOUT=200

if [ $DEVICE_NUM -lt 2 ] && [ $DEVICE_NUM -gt 8 ]
then
    echo "error: DEVICE_NUM=$DEVICE_NUM is not in (2-8)"
exit 1
fi

if [ $SCALE -ne 2 ] && [ $SCALE -ne 4 ]
then
    echo "error: SCALE=$SCALE is not 2 or 4."
exit 1
fi

if [ ! -d "$3" ]; then
    echo "error: DATASET_PATH:$3 does not exist"
    exit 1
fi

rm -rf ./train_distribute
mkdir ./train_distribute
cp -r ../src ./train_distribute
cp -r ../*.py ./train_distribute

echo "start distribute training"
env > env.log

if [ $# == 3 ]
then
    nohup mpirun -n $DEVICE_NUM --allow-run-as-root --output-filename ./train_distribute/log_output --merge-stderr-to-stdout \
    python ../train.py --run_distribute True \
        --device_num $DEVICE_NUM \
        --scale $SCALE \
        --dataset_GT_path $3 \
        --device_target GPU \
        > ../train_dis.log 2>&1 &
fi
cd ..
