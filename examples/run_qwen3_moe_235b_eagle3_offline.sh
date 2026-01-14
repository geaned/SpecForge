#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for qwen3
NUM_GPUS=${1:-8}
# MODEL_PATH=Qwen/Qwen3-235B-A22B-Instruct-2507
MODEL_PATH=Qwen/Qwen3-235B-A22B-Instruct-2507
DATASET_PATH=
HIDDEN_STATES_DIR=

# 0 - basic functionality
# 1 - fills data points with random noise of max length (for perf testing)
TESTING=0 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --data-path $DATASET_PATH \
    --model-path $MODEL_PATH \
    --cache-dir $ROOT_DIR/cache \
    --output-path $HIDDEN_STATES_DIR \
    --chat-template yagpt_custom \
    --max-length 10240 \
    --enable-aux-hidden-states \
    --tp-size $NUM_GPUS \
    --batch-size 4 \
    --mem-fraction-static 0.75 \
    --batch-size-to-save 32

TESTING=0 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/yagpt-eagle3.json \
    --train-data-path $DATASET_PATH \
    --train-hidden-states-path $HIDDEN_STATES_DIR \
    --output-dir $ROOT_DIR/yagpt-eagle \
    --num-epochs 2 \
    --draft-global-batch-size 4 \
    --draft-micro-batch-size 4 \
    --learning-rate 5e-5 \
    --warmup-ratio 5e-3 \
    --max-length 10240 \
    --chat-template yagpt_custom \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --draft-attention-backend flex_attention \
    --ttt-length 3 \
    --report-to tensorboard \
    --build-dataset-num-proc 32 \
    --log-steps 1 \
    --save-strategy steps \
    --save-interval 10000
