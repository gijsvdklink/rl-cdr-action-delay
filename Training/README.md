# Training

Each model is trained for 60,000 episodes. Rollouts of 2,048 steps are used as full batches for 30 epochs. Observations and rewards are normalised.

Commands are run from the repository root and repeated with seeds 1–7.

```bash
python Training/train.py --delay none --seed 1
python Training/train.py --delay deterministic --delay-mean 30 --seed 1
python Training/train.py --delay lognormal --delay-mean 30 --seed 1
```

Checkpoints are saved under `Models/<condition>/<condition>_seed<seed>/`. Each model must be retained with its matching `_vecnorm.pkl` file. `final_model` is also saved after interruption.

On resume, the episode counter is reset. The remaining count must be supplied, for example after 41,000 episodes.

```bash
python Training/train.py --delay none --seed 1 --resume --max-episodes 19000
```

Completion must be verified in the console output. A ceiling of 300 million cumulative steps is applied.
