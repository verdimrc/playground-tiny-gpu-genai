#!/bin/bash

set -exuo pipefail

declare -a MCORE_ARGS=(
    --json-keys title question answer
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model /tokenizer/tokenizer.model
    --append-eod
    --workers 8
)

declare -a DOCKER_ARGS=(
    --user $(id -u)
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "$(pwd)/law-qa-curated-v3-07212024-withtags-nosynth:/data"
    -v "$(pwd)/llama2-tokenizer:/tokenizer"
    -e HF_HUB_DISABLE_TELEMETRY=1
    -e HF_HUB_OFFLINE=1
)

DOCKER_IMAGE=nvcr.io/nvidia/nemo:24.05.01

/usr/bin/time docker run -it --rm "${DOCKER_ARGS[@]}" $DOCKER_IMAGE \
    python3 /opt/megatron-lm/tools/preprocess_data.py \
    --input /data/law-qa-train.jsonl \
    --output-prefix /data/train/law-qa-train \
    "${MCORE_ARGS[@]}"

/usr/bin/time docker run -it --rm "${DOCKER_ARGS[@]}" $DOCKER_IMAGE \
    python3 /opt/megatron-lm/tools/preprocess_data.py \
    --input /data/law-qa-val.jsonl \
    --output-prefix /data/val/law-qa-val \
    "${MCORE_ARGS[@]}"

/usr/bin/time docker run -it --rm "${DOCKER_ARGS[@]}" $DOCKER_IMAGE \
    python3 /opt/megatron-lm/tools/preprocess_data.py \
    --input /data/law-qa-test.jsonl \
    --output-prefix /data/test/law-qa-test \
    "${MCORE_ARGS[@]}"
