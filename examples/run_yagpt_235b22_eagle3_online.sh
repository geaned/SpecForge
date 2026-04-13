#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for yagpt
# Note: --dist-timeout 120 (minutes) prevents NCCL timeout during checkpoint save
NUM_GPUS=${1:-8}
MODEL_PATH=/home/geaned/models/hf/yandexgpt-5.1-235b22-202512
DATASET_PATH=/home/geaned/scripts/train_new.tsv

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --dist-timeout 120 \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/yagpt-eagle3-235b22.json \
    --train-data-path $DATASET_PATH \
    --output-dir $ROOT_DIR/yandexgpt-eagle-new \
    --num-epochs 1 \
    --draft-global-batch-size 4 \
    --draft-micro-batch-size 1 \
    --learning-rate 1e-4 \
    --warmup-ratio 5e-3 \
    --max-length 20480 \
    --chat-template yagpt_custom \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --attention-backend flex_attention \
    --ttt-length 3 \
    --report-to tensorboard \
    --build-dataset-num-proc 32 \
    --log-steps 1 \
    --save-strategy steps \
    --save-interval 2500
