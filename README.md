# DC-GRPO

## Repository layout

- `dcgrpo.py` - Implementation of DC-GRPO: dynamic clipping bound computation, DC-GRPO loss, and the `DCGRPOTrainer` class.
- `train.py` - Training entry point: loads models, tokenizer, dataset, and launches training using `DCGRPOTrainer`.
- `testing_util.py` - Utilities for evaluating generated code: runs generated code against input/output pairs and returns pass/fail results.
- `grpo_data/` - Dataset directory saved in `datasets` format (contains `data-00000-of-00001.arrow` and `dataset_info.json`).

---

## Data format

Dataset location: `grpo_data/`

- Format: `datasets` serialized archive (arrow file + metadata). See `dataset_info.json` for details.
- Fields (from `dataset_info.json`):
  - `task_id` (string)
  - `prompt` (list of {role, content})
  - `completion` (list of {role, content})

- According to the included metadata, the train split has ~1556 examples.

To replace or expand the dataset, build a `Dataset`/`DatasetDict` and call `save_to_disk("./grpo_data")`.

---

## Quick start (training)

After installing dependencies and preparing the dataset in `./grpo_data`, start training:

```
python train.py --model Qwen --task <your_task_name>
```

Script arguments (as used in `train.py`):

- `--model` : model key to use for generation (the script maps several keys to HF model IDs: `Qwen`, `DeepSeek`, `Llama`).
- `--task` : name used for the output folder. Models are saved under `./saved_models/{task}/{model}-DCGRPO`.

Notes about the training script:

- The dataset is loaded from disk via `datasets.load_from_disk("./grpo_data")`.
- A reward function `efficiency_reward` is defined and uses `testing_util.run_test` to run generated code against stored test cases to compute a pass-rate reward.
- Training arguments are provided via `GRPOConfig` in `train.py` (bf16, batch size, num_epochs, num_generations, vLLM settings, etc.).

---