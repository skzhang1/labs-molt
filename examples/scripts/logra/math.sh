#!/usr/bin/env bash
# Single-node Math: 4 FSDP training GPUs + 4 vLLM generation GPUs.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model checkpoint}"
PROMPT_DATASET="${PROMPT_DATASET:?Set PROMPT_DATASET to the DAPO-Math7500 training split}"
EVAL_DATASET="${EVAL_DATASET:?Set EVAL_DATASET to the MATH-500 evaluation split}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT to a new output directory outside this repository}"
SEED="${SEED:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MAX_AGENT_TURNS=1 TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$SAVE_ROOT"
cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --actor.optim logra --actor.loss_mode logra --algo.advantage.estimator logra \
  --actor.logra.rank 256 --actor.logra.optimizer rowadam \
  --actor.logra.kl_mode auto --actor.logra.target_norm 6e-5 \
  --actor.logra.kl_budget 2e-4 --actor.logra.kl_budget_final 2e-5 \
  --actor.logra.kl_budget_steps 300 \
  --actor.adam.betas 0.9 0.95 --actor.adam.eps 1e-8 --actor.adam.weight_decay 0 \
  --data.prompt_dataset "$PROMPT_DATASET" --data.input_key prompt --data.label_key reward_model \
  --data.apply_chat_template --data.max_samples 7500 --data.max_len 4096 \
  --rollout.batch_size 128 --rollout.n_samples_per_prompt 1 --rollout.micro_batch_size 1 \
  --rollout.vllm_generate_batch_size 32 --rollout.temperature 1 --rollout.top_p 1 \
  --train.seed "$SEED" --train.max_updates 300 --train.num_episodes 1 --train.max_epochs 1 \
  --train.batch_size 128 --train.micro_batch_size 1 --train.async_queue_size 1 \
  --train.force_sync_mode --train.colocate_fsdp_models \
  --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
  --vllm.num_engines 4 --vllm.tensor_parallel_size 1 --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.85 --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation flash_attention_2 \
  --fsdp.tp_size 1 --fsdp.ep_size 1 --fsdp.cp_size 1 --fsdp.offload none \
  --actor.gradient_checkpoint full --actor.entropy_coef 0 --actor.max_norm 0 \
  --algo.advantage.is_correction_level token --algo.advantage.is_correction_mode trunc \
  --algo.advantage.is_correction_threshold 0.5 2.0 --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 --train.agent_path "$REPO_ROOT/examples/python/agents/math.py" \
  --eval.dataset "$EVAL_DATASET" --eval.steps 5 --eval.n_samples_per_prompt 4 \
  --eval.temperature 0.7 --eval.top_p 0.7 --eval.eval_at_start \
  --ckpt.output_dir "$SAVE_ROOT/hf" --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 25 --ckpt.max_num 2 --ckpt.best_metric_key none \
  --logger.logging_steps 1 --logger.tensorboard_dir "$SAVE_ROOT/tb" "$@"
