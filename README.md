# Action-response delays in RL-based conflict detection and resolution

The effect of controller response delays on a PPO conflict detection and resolution advisory tool is investigated in a simulated airspace sector. Traffic is generated within a random sector and simulated in BlueSky. Every five simulated seconds, an aircraft is selected for a possible heading or speed advisory, which is then executed after the configured response delay.

The environment is accessed through Gymnasium, while policies are trained with the adjusted PPO configuration in Stable Baselines3. Observation and reward normalisation are applied during training, and the normalisation statistics are saved with each model.

- `Environment/` — Gymnasium environment, described in [ENVIRONMENT.md](ENVIRONMENT.md).
- `Training/` — adjusted PPO, described in [Training/README.md](Training/README.md).
- `Validation/` — [dense 30-aircraft experiment](Validation/README.md).
- `Visualisation/` — simple pygame view of the sector and aircraft.

## Setup

Python 3.11 is required. The environment can be set up with the following commands, run from this folder.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Once installed, the setup can be checked with a short training run.

```bash
python Training/train.py --delay none --seed 1 --max-episodes 2 --runs-root Models/smoke
```

Navigation data may be downloaded on first use.

## Visualisation

The sector and aircraft can be viewed after pygame has been installed.

```bash
python -m pip install pygame==2.3.0
python Visualisation/visualise.py
```

The hold action is used when no model is supplied. To view a trained policy, its checkpoint can be loaded as shown below. The matching normalisation file must be stored alongside it.

```bash
python Visualisation/visualise.py --model Models/none/none_seed1/final_model.zip
```

For delayed models, the matching condition is selected with `--delay deterministic` or `--delay lognormal`, and `--delay-mean 30`. Playback can be paused with Space and restarted in a new episode with R. The window is closed with Escape. Each frame represents five simulated seconds, with playback speed set through `--fps`.
