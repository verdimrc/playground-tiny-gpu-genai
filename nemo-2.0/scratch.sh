#!/bin/bash

####
# NOTES:
# 20241114:
# - red flags: gpt2 tokenizer? Downloaded from amazon s3.
# WARNING:
# nemo.collections.llm.api.pretrain/0 [default0]:[NeMo I 2024-11-14 10:04:39 tokenizer_utils:223] Getting Megatron tokenizer for pretrained model name: megatron-gpt-345m, custom vocab file: None, and merges file: None
# nemo.collections.llm.api.pretrain/0 [default0]:[NeMo I 2024-11-14 10:04:39 tokenizer_utils:129] Getting HuggingFace AutoTokenizer with pretrained_model_name: gpt2, vocab_file: /tmp/home-haha/.cache/torch/megatron/megatron-gpt-345m_vocab, merges_files: /tmp/home-haha/.cache/torch/megatron/megatron-gpt-345m_merges, special_tokens_dict: {}, and use_fast: False

# FATAL ERROR: occur when pretraining.py does not set context parallelism.
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:   File "/opt/megatron-lm/megatron/core/parallel_state.py", line 532, in initialize_model_parallel
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:     raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]: RuntimeError: world_size (1) is not divisible by 2

## ERROR: looks like we must run as privileged container.
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:   File "/usr/local/lib/python3.10/dist-packages/torch/_inductor/runtime/runtime_utils.py", line 136, in cache_dir
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:     sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:   File "/usr/lib/python3.10/getpass.py", line 169, in getuser
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]:     return pwd.getpwuid(os.getuid())[0]
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]: torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
# nemo.collections.llm.api.pretrain/0 [default0]:[rank0]: KeyError: 'getpwuid(): uid not found: 1000'
# nemo.collections.llm.api.pretrain/0 [default0]:

####
# Treat this file as a doc, not as an executable. The extension .sh is for syntax highlighting.
#
# https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html

docker pull docker pull nvcr.io/nvidia/nemo:24.09

    ## Will cause nemo train to fail (see above, exception from torch dynamo)
    # -u $(id -u):$(id -g) \
docker run \
    -it \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/haha \
    nvcr.io/nvidia/nemo:24.09 \
    /bin/bash

########
# Inside container

mkdir -p /workspace/nemo-run
cd /workspace/nemo-run

mkdir /tmp/checkpoints  # For unprivileged container

python -c 'import nemo; print(nemo.__file__)'
#/opt/NeMo/nemo/__init__.py

# Let's get the exact commit id in the container.
   ## Fix git fatal error under unprivileged container.
   mkdir /tmp/home-haha
   export HOME=/tmp/home-haha
   git config --global --add safe.directory /opt/NeMo
( cd /opt/NeMo ; git rev-parse HEAD )
#33ccb6eca9c76386544cd3683275f5d846abdf82

ln -s /haha/llama3_pretraining.py /workspace/nemo-run/


# Clear previous incomplete checkpoints, to avoid:
#
# nemo.collections.llm.api.pretrain/0 [default0]:ValueError: Last checkpoint is unfinished and cannot be used to resume the training. Please remove the checkpoint manually to avoid unexpected cosequences, such as restarting from scratch. Hint: Iteration number can be added to the checkpoint name pattern to maximize chance that there is at least one finished last checkpoint to resume from.
rm -fr /tmp/checkpoints/llama3/
python llama3_pretraining.py


####
# https://github.com/NVIDIA/NeMo/blob/33ccb6eca9c76386544cd3683275f5d846abdf82/nemo/collections/llm/recipes/llama3_8b.py#L173
#
# In the end, press n (otherwise, y means to launch, and without preparation, may fail -- expected).
nemo llm pretrain --factory llama3_8b
nemo llm train    --factory llama3_8b  # Error: no factory
nemo llm finetune --factory llama3_8b


######
huggingface-cli login
# then, provide HF credential.

# python:
from nemo.lightning.io import import_ckpt
from nemo.collections import llm
# Below need to get HF approval to the model.
imported_path = import_ckpt(llm.LlamaModel(llm.Llama3Config8B), "hf://meta-llama/Meta-Llama-3.1-8B-Instruct")
# Killed. DUnno why.

##python /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py
