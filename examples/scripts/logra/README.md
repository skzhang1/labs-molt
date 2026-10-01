# LOGRA Math training

Install Molt using the repository installation instructions. Run on one node with
8 GPUs (4 training + 4 generation) and 128 CPUs. Supply the model and datasets
separately; no data is included.

```bash
MODEL_PATH=/path/to/Qwen2.5-Math-1.5B \
PROMPT_DATASET=/path/to/dapo_math7500/train \
EVAL_DATASET=/path/to/math500/eval \
SAVE_ROOT=/path/to/outputs/logra-s1 SEED=1 \
bash examples/scripts/logra/math.sh
```

The recipe runs 300 updates with 128 prompts × 1 response, rank 256, RowAdam,
predicted-KL budget annealing, mismatch subtraction, and 4096-token context.
Evaluation uses four responses per question every five updates. Checkpoints,
TensorBoard events, and per-update memory/performance metrics go to `SAVE_ROOT`.
Use `SEED=2` with a separate output directory for the second seed. To resume,
use the same directory and append `--ckpt.load_enable`.
