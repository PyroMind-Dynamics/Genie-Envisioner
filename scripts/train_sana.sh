#!/usr/bin/bash

# 关闭NCCL_SHM,防止通讯超时中断训练
export NCCL_SHM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

if [ -z $WORLD_SIZE ]; then
NGPU=`nvidia-smi --list-gpus | wc -l`
echo "Training on 1 Nodes, $NGPU GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    --master-port 29501 \
    $script_path \
    --config_file $config_path
else
echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
NGPU=`nvidia-smi --list-gpus | wc -l`
torchrun --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
fi
