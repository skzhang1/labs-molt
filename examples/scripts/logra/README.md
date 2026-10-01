# LOGRA Math training

Install Molt using the repository installation instructions. Run on one node with
8 GPUs (4 training + 4 generation) and 128 CPUs. Model weights and datasets are
stored separately from this repository.

## Data

| Use | Public source | Selection |
| --- | --- | --- |
| Training | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | First 7,500 unique prompts from the `default` / `train` split, retaining source order |
| Evaluation | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | All 500 questions from the `test` split |

Training deduplicates by `prompt[0]["content"]` before taking 7,500 examples; this
is not a randomly sampled subset. Keep the DAPO prompts and reward labels as
provided. For MATH-500, wrap `problem` as a user message and use `answer` as the
string-valued ground truth. Do not include reference solutions in the prompts.

Run the following in the Molt Python environment to download and prepare the
splits in Hugging Face `save_to_disk` format. The generic `prepare_dapo.py`
uses a different evaluation dataset by default and does not convert MATH-500.
The revisions below pin the public sources used for this recipe.

```bash
export DATA_ROOT=/path/to/data/dapo_math7500_math500
python - <<'PY'
import os
from pathlib import Path
from datasets import Dataset, load_dataset

root = Path(os.environ["DATA_ROOT"])
assert not (root / "train").exists() and not (root / "eval").exists()
train = load_dataset(
    "BytedTsinghua-SIA/DAPO-Math-17k", "default", split="train",
    revision="65877096c24ffa7abc4e4fa5edb95cf3413a5674", streaming=True,
)
seen, rows = set(), []
for row in train:
    prompt = row["prompt"][0]["content"]
    if prompt in seen:
        continue
    seen.add(prompt)
    rows.append({"datasource": row.get("data_source", "dapo_math_17k"),
                 "prompt": row["prompt"], "reward_model": row["reward_model"]})
    if len(rows) == 7500:
        break
assert len(rows) == 7500
Dataset.from_list(rows).save_to_disk(str(root / "train"))

evaluation = load_dataset(
    "HuggingFaceH4/MATH-500", split="test",
    revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
)
assert len(evaluation) == 500
Dataset.from_list([
    {"datasource": "math500",
     "prompt": [{"role": "user", "content": row["problem"]}],
     "reward_model": {"style": "rule", "ground_truth": str(row["answer"])}}
    for row in evaluation
]).save_to_disk(str(root / "eval"))
PY
```

## Train

Download [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
and set `MODEL_PATH` to its local directory.

```bash
MODEL_PATH=/path/to/Qwen2.5-Math-1.5B \
PROMPT_DATASET="$DATA_ROOT/train" \
EVAL_DATASET="$DATA_ROOT/eval" \
SAVE_ROOT=/path/to/outputs/logra-s1 SEED=1 \
bash examples/scripts/logra/math.sh
```

The recipe runs 300 updates with 128 prompts × 1 response, rank 256, RowAdam,
predicted-KL budget annealing, mismatch subtraction, and 4096-token context.
Evaluation uses four responses per question every five updates. Checkpoints,
TensorBoard events, and per-update memory/performance metrics go to `SAVE_ROOT`.
Use `SEED=2` with a separate output directory for the second seed. To resume,
use the same directory and append `--ckpt.load_enable`.
