""" https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#execute-locally.

Changes: search for HAHA.
"""
import nemo_run as run

from nemo.collections import llm


# HAHA: nemotron -> llama3
def configure_recipe(nodes: int = 1,
                     gpus_per_node: int = 2,
                     **kwargs   # HAHA
):
    recipe = llm.llama3_8b.pretrain_recipe(
        dir="/tmp/checkpoints/llama3", # Path to store checkpoints  # HAHA: change to unpriveleged dir
        name="llama3_pretraining",
        # tensor_parallelism=2,  # HAHA: llama3_8b recipe doesn't have this.
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
        # max_steps=100, # Setting a small value for the quickstart  # HAHA: N/A for llama3
        **kwargs,  # HAHA
    )

    recipe.trainer.val_check_interval = 100

    ## From the animation: example to override activation function.
    ## Not needed for llama3. Left here just for show off.
    #import torch.nn.functional as F
    #recipe.model.config.activation_func = F.silu

    return recipe

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

def run_pretraining():
    """HAHA: tiny model on 1 GPU. See https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#change-the-number-of-gpus"""
    #import pdb; pdb.set_trace()   # b /opt/NeMo/nemo/collections/llm/recipes/llama3_8b.py:199
    recipe = configure_recipe(
        # HAHA: braveheart.
        # https://github.com/NVIDIA/NeMo/blob/33ccb6eca9c76386544cd3683275f5d846abdf82/nemo/collections/llm/recipes/llama3_8b.py#L152
        performance_mode=True
    )
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    # Change to 1 GPU

    # Change executor params
    executor.ntasks_per_node = 1
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    # Change recipe params

    # The default number of layers comes from the recipe in nemo where num_layers is 32
    # Ref: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/gpt/model/nemotron.py
    # To run on 1 GPU without TP, we can reduce the number of layers to 8 by setting recipe.model.config.num_layers = 8
    # HAHA: run on laptop: RTX A1000 (6GB)
    recipe.model.config.num_layers = 1
    recipe.model.config.hidden_size = 256
    recipe.model.config.ffn_hidden_size=512
    #recipe.model.config.num_attention_heads=4  # HAHA: error: not divisible by num_gqa_heads.
    recipe.model.config.num_attention_heads=32 # HAHA: could not find ways to change the gqa heads in nemo codebase. Exception shows this happens down in the mcore codebase.
    # We also need to set TP to 1, since we had used 2 for 2 GPUs.
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    # Lastly, we need to set devices to 1 in the trainer.
    recipe.trainer.devices = 1

    # HAHA: more tweaks. Can quickly figure out the namespace with: nemo llm pretrain --factory llama3_8b
    #### https://github.com/NVIDIA/NeMo/blob/33ccb6eca9c76386544cd3683275f5d846abdf82/nemo/collections/llm/gpt/model/llama.py#L130
    #### - Llama3 8b seq length = 8k.
    #### - llama3.1 8b seq length = 128k
    recipe.model.config.seq_length = 128
    recipe.data.seq_length = recipe.model.config.seq_length
    recipe.data.global_batch_size=64
    recipe.trainer.max_steps = 2
    recipe.trainer.val_check_interval = 30  # HAHA: already the default.
    recipe.trainer.strategy.context_parallel_size=1  # HAHA: needed to avoid fatal error. See scratch.sh for the error.
    print(recipe)   #; return
    run.run(recipe, executor=executor)

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()
