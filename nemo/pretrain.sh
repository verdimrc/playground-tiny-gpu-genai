#!/bin/bash

set -exuo pipefail

##################################################
###### Model architectures (example presets) #####
##################################################
# Feel free to choose one of the sample presents, or completely define your own
# custom model size.

## Test llama (tiny)
declare -a MEGATRON_ARGS=(
   --num-layers 4
   --hidden-size 512
   --num-attention-heads 4

   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
)

## llama2-7b-hf
#declare -a MEGATRON_ARGS=(
#    --num-layers 32
#    --hidden-size 4096
#    --num-attention-heads 32
#
#    --tensor-model-parallel-size 1
#    --pipeline-model-parallel-size 1
#)

## llama2-13b-hf
#declare -a MEGATRON_ARGS=(
#    --num-layers 40
#    --hidden-size 5120
#    --num-attention-heads 40
#
#    --tensor-model-parallel-size 2
#    --pipeline-model-parallel-size 1
#    --sequence-parallel
#
#    --use-distributed-optimizer
#    --overlap-grad-reduce
#    --overlap-param-gather
#)

# llama2-70b-hf.
#declare -a MEGATRON_ARGS=(
#    --num-layers 80
#    --hidden-size 8192
#    --num-attention-heads 64
#    --group-query-attention
#    --num-query-groups 8
#
#    --tensor-model-parallel-size 4
#    --pipeline-model-parallel-size 4
#    --sequence-parallel
#
#    --use-distributed-optimizer
#    --overlap-grad-reduce
#    --overlap-param-gather
#)

# Required for Llama2-style architecture. Do not comment or remove.
MEGATRON_ARGS+=(
   --untie-embeddings-and-output-weights
   --position-embedding-type rope
   --no-position-embedding
   --normalization RMSNorm
   --swiglu
   --no-masked-softmax-fusion
)

# Additional flags to make it possible to test with as few nodes as possible
MEGATRON_ARGS+=(
    --no-rope-fusion
    --use-flash-attn
    --transformer-impl transformer_engine
)


###########################
###### User Variables #####
###########################

: "${SEQ_LENGTH:=4096}"
: "${MAX_POSITION_EMBEDDINGS:=4096}"
: "${MICRO_BATCH_SIZE:=1}"
: "${GLOBAL_BATCH_SIZE:=2048}"

# default variables for Enroot
: "${IMAGE:=$(pwd)/megatron-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${FSX_MOUNT:=$(pwd):$DATA_PATH}"


#########################
## Command and Options ##
#########################
declare -a TORCHRUN_ARGS=(
    # change this to match the number of gpus per node:
    --nproc_per_node=1
    --nnodes=1
    --standalone
    #--rdzv_id=HAHA
    #--rdzv_backend=c10d
    #--rdzv_endpoint=$(hostname)
)

MEGATRON_ARGS+=(
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE

    # Example how to control training duration using steps rather than number of samples.
    --train-iters 5

    # Example how to disable all validations, hence only training steps performed.
    --split 100,0,0
)

declare -a DOCKER_ARGS=(
    # Bah, unprivileged mode doesn't work ootb. Got this error:
    #
    #   File "/usr/lib/python3.10/getpass.py", line 169, in getuser
    #     return pwd.getpwuid(os.getuid())[0]
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # KeyError: 'getpwuid(): uid not found: 1000'
    #
    #--user $(id -u)

    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "$(pwd)/law-qa-curated-v3-07212024-withtags-nosynth:/data"
    -v "$(pwd)/llama2-tokenizer:/tokenizer"
    -e HF_HUB_DISABLE_TELEMETRY=1
    -e HF_HUB_OFFLINE=1
    -e NCCL_ASYNC_ERROR_HANDLING=1
    #-e NCCL_DEBUG=INFO
    -e NCCL_AVOID_RECORD_STREAMS=1          # torch<2.2
    -e TORCH_NCCL_AVOID_RECORD_STREAMS=1    # torch>=2.2

    # async runtime error ...
    -e CUDA_DEVICE_MAX_CONNECTIONS=1
)

DOCKER_IMAGE=nvcr.io/nvidia/nemo:24.05.01

/usr/bin/time docker run -it --rm "${DOCKER_ARGS[@]}" $DOCKER_IMAGE \
    python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" /opt/megatron-lm/pretrain_gpt.py \
        "${MEGATRON_ARGS[@]}" \
        --use-mcore-models \
        --log-throughput \
        --lr 6.0e-5 \
        --min-lr 6.0e-6 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 0 \
        --data-path /data/train/law-qa-train_answer_document \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model /tokenizer/tokenizer.model \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --fp16
