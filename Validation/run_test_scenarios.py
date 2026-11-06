"""Evaluate trained policies in the dense 30-aircraft experiment.

Traffic density is fixed at 0.0001 aircraft/km². Each policy is evaluated on
500 held-out test scenarios across nine delay conditions. Seven training seeds are
used per model type, with one additional no-CR baseline (190 runs in total).
Observation normalisation is frozen and actions are selected deterministically.
Completed run files are skipped when the same command is repeated.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
import json
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MAIN_DIR)

from Environment.config import (CONFIG, HOLD_ACTION, OBS_OWNSHIP_LABELS, TRAINING_SEEDS,
                                VALIDATION_SEEDS)

MODELS_DIR  = os.path.join(MAIN_DIR, 'Models')
RESULTS_DIR = os.path.join(MAIN_DIR, 'Results', 'dense_30ac')

RESPONSE_STATE = [OBS_OWNSHIP_LABELS.index(name) for name in ('pending', 'wait_s', 'pend_hdg',
                                                              'pend_spd')]

MODEL_TYPES = {'none': 'none', 'deterministic': 'deterministic_30s', 'lognormal': 'lognormal_30s'}
PREFIX      = {'none': 'N', 'deterministic': 'D', 'lognormal': 'L'}
NO_CR       = 'no_cr'

NO_DELAY     = ('none', 0)
TEST_DELAYS  = [NO_DELAY] + [(law, mean) for law in ('deterministic', 'lognormal')
                             for mean in (15, 30, 45, 60)]
TERMINALS    = 1
WORKERS      = 4
PRINT_EVERY  = 25


def all_runs():
    """Every test run as (model type, seed, delay distribution, mean delay): 3 x 7 x 9 + no CR = 190."""
    runs = [(model_type, seed, law, mean) for model_type in MODEL_TYPES
            for seed in TRAINING_SEEDS for law, mean in TEST_DELAYS]
    return runs + [(NO_CR, 0) + NO_DELAY]


def terminal_share(terminal, terminals):
    """Return the experiment subset assigned to the specified terminal."""
    return all_runs()[terminal - 1::terminals]


def model_name(model_type, seed):
    """The short model name used in the results, e.g. L3 or NoCR."""
    return 'NoCR' if model_type == NO_CR else f'{PREFIX[model_type]}{seed}'


def run_path(out, model_type, seed, law, mean):
    """Construct the CSV output path for one evaluation run."""
    test = 'no_delay' if law == 'none' else f'{law}_{mean}s'
    return os.path.join(out, 'runs', f'{model_name(model_type, seed)}_{test}.csv')


def checkpoint_steps(path):
    """Read the cumulative training step count from a checkpoint."""
    with zipfile.ZipFile(path) as archive:
        return int(json.loads(archive.read('data').decode())['num_timesteps'])


def find_model(models, model_type, seed):
    """Return the checkpoint with the highest cumulative training step count."""
    folder = MODEL_TYPES[model_type]
    run_dir = os.path.join(models, folder, f'{folder}_seed{seed}')
    found = [(checkpoint_steps(path), path)
             for path in (os.path.join(run_dir, name) for name in ('final_model.zip', 'last_model.zip'))
             if os.path.exists(path)]
    if not found:
        sys.exit(f'no model in {run_dir}')
    return max(found)


class _Ignored:
    def __init__(self, *args):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    """Load observation statistics with compatibility handling for NumPy module paths.
    Serialised NumPy random states are omitted because evaluation does not use them."""

    def find_class(self, module, name):
        if module.startswith('numpy.random'):
            return _Ignored
        if np.__version__.startswith('1.'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def load_policy(path):
    """Load a policy checkpoint and its frozen observation normalisation statistics."""
    from stable_baselines3 import PPO

    with open(path.replace('.zip', '_vecnorm.pkl'), 'rb') as handle:
        normaliser = _Unpickler(handle).load()
    mean = np.asarray(normaliser.obs_rms.mean, dtype=np.float32)
    std = np.sqrt(np.asarray(normaliser.obs_rms.var, dtype=np.float32) + normaliser.epsilon)
    return PPO.load(path, device='cpu'), mean, std, float(normaliser.clip_obs)


def choose(policy, observation, blind=False):
    """Select a deterministic action, optionally masking response-state features for no-delay policies."""
    if policy is None:
        return HOLD_ACTION
    if blind:
        observation = observation.copy()
        observation[RESPONSE_STATE] = 0.0
    model, mean, std, clip = policy
    action, _ = model.predict(np.clip((observation - mean) / std, -clip, clip), deterministic=True)
    return int(np.asarray(action).flat[0])


def count_executed(atco):
    """Count executed heading and speed instructions. Pending advisories may be replaced
    before execution, so issued and executed instruction counts are recorded separately."""
    executed = {'turns': 0, 'speed_changes': 0}
    act_if_ready = atco.act_if_ready

    def counted(*args):
        ready = act_if_ready(*args)
        if ready is not None:
            executed['speed_changes' if 'target_mach' in ready[1] else 'turns'] += 1
        return ready

    atco.act_if_ready = counted
    return executed


def fly(env, policy, scenario_seed, blind=False):
    """Evaluate one scenario and return episode metrics and flight-hour-normalised rates."""
    observation, _ = env.reset(options={'scenario_seed': scenario_seed})
    executed = count_executed(env.atco)
    while True:
        observation, _, _, truncated, info = env.step(choose(policy, observation, blind))
        if truncated:
            kpis = {key.removeprefix('ep_'): value for key, value in info.items()
                    if key.startswith('ep_')}
            executed['advisories'] = executed['turns'] + executed['speed_changes']
            for kind, count in executed.items():
                kpis[f'{kind}_executed'] = count
                kpis[f'{kind}_executed_per_fh'] = count / kpis['flight_hours']
            return kpis


def test_run(model_type, seed, law, mean, models, out, episodes):
    """Evaluate one policy under a specified delay condition across held-out test scenarios."""
    from Environment import AirspaceEnv

    path = run_path(out, model_type, seed, law, mean)
    if os.path.exists(path):
        print(f'{os.path.basename(path)} already done', flush=True)
        return

    if model_type == NO_CR:
        policy, steps = None, 0
    else:
        steps, checkpoint = find_model(models, model_type, seed)
        policy = load_policy(checkpoint)
    env = AirspaceEnv(delay_mode=law, delay_mean_s=float(mean)) if mean else AirspaceEnv(delay_mode='none')

    scenarios = VALIDATION_SEEDS[:episodes]
    print(f'{model_name(model_type, seed)} ({steps:,} steps) in {law} {mean} s: '
          f'{len(scenarios)} scenarios', flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    started = time.time()
    with open(path + '.part', 'w', newline='') as handle:
        writer = None
        for scenario, scenario_seed in enumerate(scenarios):
            kpis = fly(env, policy, scenario_seed, blind=model_type == 'none')
            row = {'model': model_name(model_type, seed), 'train_type': model_type, 'seed': seed,
                   'model_steps': steps, 'test_type': law, 'test_delay': mean,
                   'scenario': scenario, 'scenario_seed': scenario_seed,
                   'n_aircraft': env.n_aircraft, 'rho': env.rho, **kpis}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            done = scenario + 1
            if done % PRINT_EVERY == 0 or done == len(scenarios):
                rate = (time.time() - started) / done
                print(f'  {os.path.basename(path)[:-4]:<28} {done:>3}/{len(scenarios)}  '
                      f'{rate:4.1f} s/scenario  {rate * (len(scenarios) - done) / 60:5.1f} min to go',
                      flush=True)
    os.replace(path + '.part', path)


def _init_worker(n_ac, density):
    import torch
    torch.set_num_threads(1)
    if n_ac is not None:
        CONFIG['n_aircraft'] = lambda rng: n_ac
    if density is not None:
        CONFIG['rho'] = lambda rng: density


def _test_run_safely(run, models, out, episodes):
    try:
        test_run(*run, models, out, episodes)
        return None
    except (Exception, SystemExit) as error:
        return f'{os.path.basename(run_path(out, *run))}: {error!r}'


def run_pool(terminal, terminals, workers, models, out, episodes, n_ac, density):
    """Evaluate the assigned runs using separate worker processes, each with one environment."""
    share = [run for run in terminal_share(terminal, terminals)
             if not os.path.exists(run_path(out, *run))]
    print(f'share {terminal} of {terminals}: {len(share)} runs to do, {workers} at a time -> {out}',
          flush=True)
    failed = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(n_ac, density)) as pool:
        for error in pool.map(_test_run_safely, share, repeat(models), repeat(out), repeat(episodes)):
            if error:
                failed.append(error)
    print('all runs finished' + ('; failed:\n  ' + '\n  '.join(failed) if failed else ''), flush=True)
    if failed:
        sys.exit(1)


def merge(out):
    """Combine the per-run CSVs into one, once every run has finished."""
    missing = [os.path.basename(path) for path in (run_path(out, *run) for run in all_runs())
               if not os.path.exists(path)]
    if missing:
        sys.exit(f'{len(missing)} runs are not finished yet:\n  ' + '\n  '.join(missing))

    path = os.path.join(out, 'test_results.csv')
    # Results are combined in experiment order after every run has completed.
    with open(path + '.part', 'w', newline='') as target:
        writer = None
        for run in all_runs():
            with open(run_path(out, *run), newline='') as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    os.replace(path + '.part', path)
    print(f'Combined results saved to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--terminal', type=int, default=1,
                        help='which share of the runs, 1 to --terminals (default 1)')
    parser.add_argument('--terminals', type=int, default=TERMINALS,
                        help=f'into how many shares the runs are split, e.g. one per machine '
                             f'(default {TERMINALS}: all runs)')
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help=f'runs flown side by side, each in its own process (default {WORKERS})')
    parser.add_argument('--merge', action='store_true', help='combine all runs into one CSV')
    parser.add_argument('--run', nargs=4, metavar=('TYPE', 'SEED', 'LAW', 'MEAN'),
                        help='a single test run, e.g. --run lognormal 3 deterministic 45')
    parser.add_argument('--models', default=MODELS_DIR, help=f'default {MODELS_DIR}')
    parser.add_argument('--out', default=RESULTS_DIR, help=f'default {RESULTS_DIR}')
    parser.add_argument('--episodes', type=int, default=len(VALIDATION_SEEDS),
                        help='test scenarios per run, from the start of the set')
    args = parser.parse_args()
    out, models = os.path.abspath(args.out), os.path.abspath(args.models)

    if not 1 <= args.episodes <= len(VALIDATION_SEEDS):
        parser.error('--episodes must be between 1 and 500')
    if args.workers < 1 or not 1 <= args.terminal <= args.terminals:
        parser.error('workers must be positive and terminal must be within 1..terminals')

    # Apply the fixed aircraft count and density for the dense evaluation experiment.
    _init_worker(30, 0.0001)

    if args.merge:
        merge(out)
    elif args.run:
        model_type, seed, law, mean = args.run
        test_run(model_type, int(seed), law, int(mean), models, out, args.episodes)
    else:
        run_pool(args.terminal, args.terminals, args.workers, models, out, args.episodes,
                 30, 0.0001)


if __name__ == '__main__':
    main()
