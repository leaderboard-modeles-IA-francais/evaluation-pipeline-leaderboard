# Project README

## Overview

This project is dedicated to disclosing the essential tools and scripts for executing a ClearML task that results in the execution of benchmarks published under HuggingFace space: fr-gouv-coordination-ia/llm_leaderboard_fr. This project is provided solely for transparency purposes, since the datasets are held hidden.

The benchmark executions are scheduled with the clearml platform which spawns clearml-agent jobs on a cluster.

By default, executions run on 2 nodes with 2 H100 GPUs (no infiniband between the nodes) for a total duration of 1 hour and 30 minutes.

Depending on the constraints of the model, some tuning is required; the following cases are given as examples of such tuning:

``` python
# Not handled on the same cluster
model_too_large_list = ["allenai/Llama-3.1-Tulu-3-405B"]

# Not handled with vllm
model_incompatible_list = ["teapotai/teapotllm"]

# Either requiring more VRAM, or a number of nodes to divide the number of attention heads
model_nb_nodes_map = {
    "mistralai/Mistral-Large-Instruct-2411": 4,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 4,
    "eval_mistralai/Mixtral-8x22B-Instruct-v0.1": 4,
    "Qwen/Qwen2.5-Math-72B-Instruct": 4,
    "microsoft/phi-4": 1,
    "microsoft/Phi-3-medium-128k-instruct": 1,
    "jpacifico/Chocolatine-2-14B-Instruct-v2.0": 1,
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": 1,
}

# Slightly more VRAM required
model_gpu_memory_utilization_map = {
    "mistralai/Mistral-Large-Instruct-2411": 0.8,
}

model_nb_gpus_per_node_map = {

}

# Need more time to complete
model_walltime_map = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "04:30",
    "Qwen/Qwen2.5-Math-72B-Instruct": "14:00",
    "mistralai/Mistral-Large-Instruct-2411": "08:00",
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": "06:00",
    "speakleash/Bielik-11B-v2.3-Instruct": "05:00",
    "HoangHa/Pensez-v0.1-e5": "03:00",
    "MaziyarPanahi/calme-3.2-instruct-78b": "03:00"
    "arcee-ai/Virtuoso-Medium-v2": "12:00"
    "HoangHa/Pensez-Llama3.1-8B": "03:00",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "03:00",
    "baconnier/Napoleon_24B_V0.1": "08:00",
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": "24:00",
}
```

## Post-processing

The original scoring of the "bac-fr" task was weak and leading to a number of false-positive and false-negative correctness evaluation.
Instead of running all the tasks all over, a post processing step was conducted on the already generated parquet files (in "DETAILS_PATH").

``` shell
python post_precessing.py $DETAILS_PATH --scores-dir $RESULTS_PATH
```
