#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for yagpt
NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
MODEL_PATH=
DATASET_PATH=

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/yagpt-eagle3.json \
    --train-data-path $DATASET_PATH \
    --output-dir $ROOT_DIR/yagpt-eagle \
    --num-epochs 1 \
    --batch-size 1 \
    --draft-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --warmup-ratio 5e-3 \
    --max-length 10240 \
    --chat-template yagpt_custom \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend hf \
    --attention-backend flex_attention \
    --ttt-length 3 \
    --report-to tensorboard \
    --build-dataset-num-proc 32 \
    --log-interval 1 \
    --save-interval 10000
