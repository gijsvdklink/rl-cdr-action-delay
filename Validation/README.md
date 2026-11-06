# Dense 30-aircraft experiment

The trained models are evaluated with 30 aircraft at a density of 0.0001 aircraft/km². Each of the 21 models is tested on 500 held-out test scenarios in nine delay conditions. A no-CR baseline is included, giving 190 runs.

Models and matching normalisation files are read from `Models/`. The checkpoint with the highest training step count is selected. Completion of 60,000 training episodes should be verified beforehand.

From the thesis folder, the experiment is run and the results are combined as follows.

```bash
python Validation/run_test_scenarios.py
python Validation/run_test_scenarios.py --merge
```

Four worker processes are used by default. This can be changed with `--workers`. Results are saved under `Results/dense_30ac/`. Completed runs are skipped on restart. A separate `--out` folder must be used when checkpoints or the scenario count are changed.

For no-delay policies, the four response-state inputs are set to zero, as during training. Normalisation statistics are frozen during evaluation.
