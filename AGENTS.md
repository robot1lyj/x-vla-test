# Repository Guidelines



用中文回复

## Project Structure & Module Organization
- `train.py`: Accelerate-based training entry; outputs to `runnings/` by default.
- `deploy.py` (`python -m deploy`): Launches inference servers, SLURM-aware, writes connection info to `info.json`.
- `models/`: XVLA configs/modeling/processor plus action heads and transformer core.
- `datasets/`: Domain registry and loaders (`domain_handler/`), dataset config, and helpers for meta JSONs.
- `evaluation/`: Benchmark clients and task scripts per domain (see `*/README.md` and `client*.py`).
- `requirements.txt` and `README.md`: Pinned versions and usage notes.

## Environment Setup & Key Commands
- `pip install -r requirements.txt` (Python 3.10 recommended).
- Training: `accelerate launch --mixed_precision bf16 train.py --models 2toINF/X-VLA-Pt --train_metas_path /data/meta.json --output_dir runnings/exp1 [--learning_rate ...]`
- Serve: `python -m deploy --model_path checkpoints/last` to start the inference server; detects SLURM settings automatically.
- Eval: `python evaluation/simpler/WidowX/client.py --config configs/widowx.yaml` (or other clients under `evaluation/<domain>/`) to reproduce benchmarks; check each README for expected args.
- Logs: stream `train.log` in each run directory; use TensorBoard or simple tailing for quick checks.

## Coding Style & Naming Conventions
- Follow PEP8 with 4-space indents and existing type hints (`typing.Dict`, etc.).
- Modules remain `snake_case`; classes stay `CamelCase` (e.g., `XVLAProcessor`), configs named `configuration_*.py`.
- Prefer `logging` over prints and reuse helper patterns (e.g., `get_logger` in `train.py`).
- Keep helper functions small; avoid hard-coded absolute paths so configs stay portable.

## Testing & Validation
- No unit-test harness yet; validate via domain clients and dataset loaders.
- Before PRs, run at least one evaluation client end-to-end against your checkpoint and record task counts or success rates.
- For data changes, verify `datasets/domain_config.py` registration and that referenced `meta.json` paths resolve.

## Commit & Pull Request Guidelines
- Mirror existing history: short, imperative summaries (“Change default min_lr_ratio from 0.05 to 0.1”).
- PRs should state the objective, configs/flags used, logs or metric tables, and linked issues or checkpoints.
- Include screenshots or console excerpts for evaluation runs when applicable; note any new dependencies.

## Security & Assets
- Do not commit model checkpoints, raw trajectories, or credentials (HF tokens, SLURM secrets).
- Keep robot- or cluster-specific paths configurable (env vars/CLI flags) and document required env vars when adding them.
