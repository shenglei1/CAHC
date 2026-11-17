# TeamComm-owned V1 baselines

This self-contained directory runs CommNet, IC3Net, MAGIC, QMIX, TarMAC,
TeamComm, and TieComm on the same six V1 scenarios. The code does not require
or inspect an environment name. Run commands from this directory using
relative paths.

The scenarios are `hetero_mixed`, `hetero_strong`, `consensus_wide`,
`fog_sight1_level`, `dynamic_overload`, and `scale_20`. Training seeds are
`20 23 56 89 2025`.

## One-command formal experiment

```bash
python -u run_teamcomm_formal.py --scenarios hetero_mixed hetero_strong consensus_wide fog_sight1_level dynamic_overload scale_20 --baselines commnet ic3net magic qmix tarmac teamcomm tiecomm --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results --resume
```

The scheduler completes all five seeds for the current baseline and scenario
before advancing. The resume mode validates and reuses complete runs; it also
works on a new empty result directory. Add `--dry-run` to verify all 210 jobs
without training. The
seven baseline implementations retain their native models, losses, optimizers,
and action rules; only the common V1 environment, fair budget, evaluation, and
metric recording are shared.

## Reviewer 2 cross-configuration evaluation

The checkpoint-only launcher in the sibling `HyperWave-X` directory evaluates
all seven baselines together with HyperWave-X and HyperComm on the registered
`hetero_strong` source, demand-shift, and fleet-shift configurations. It reuses
the five formal checkpoints and performs no training. Pass this environment's
Python executable through `--teamcomm-python`; the generated TeamComm-owned
rows are stored under
`results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES`.
