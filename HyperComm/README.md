# HyperComm V1 experiments

This directory is self-contained. Run commands from this directory with a
compatible Python environment; the code does not require or inspect an
environment name. All paths below are relative.

## Frozen protocol

- Main scenarios: `hetero_mixed`, `hetero_strong`, `consensus_wide`,
  `fog_sight1_level`, `dynamic_overload`, `scale_20`.
- Controlled probes: `probe_complementary`, `probe_interaction`.
- Training seeds: `20 23 56 89 2025`.
- Main results: `results/FORMAL_6SCENARIOS`.
- Probe results: `results/PROBES_ORGANIZED_V1`.

## Six main scenarios

```bash
python -u run_hypercomm_formal.py --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/FORMAL_6SCENARIOS --resume
```

## Controlled probes

Probe 1 and Probe 2 use different registered evaluation seeds, so run them as
two reproducible shards below the same organized result directory.

```bash
python -u run_hypercomm_formal.py --include-probes --scenarios probe_complementary --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
python -u run_hypercomm_formal.py --include-probes --scenarios probe_interaction --train-seeds 104729 130363 155921 181081 205019 --gpu -1 --workers 5 --cpu-threads-per-job 2 --epochs 700 --episodes-per-epoch 64 --eval-seeds 31001 31002 31003 31004 31005 31006 31007 31008 31009 31010 --eval-episodes 20 --results-root ./results/PROBES_ORGANIZED_V1 --resume
```

Add `--dry-run` to any command to verify the complete five-seed schedule
without training. Paper-facing CSV files contain measured quantities and
experiment identifiers; checkpoint and execution provenance is stored in the
adjacent JSON protocol and manifest files.

## Cross-configuration generalization

The joint checkpoint-only evaluation of CAHC and all eight formal
baselines is launched from the sibling `CAHC` directory with
`run_cross_configuration_generalization.py`. It discovers the HyperComm
checkpoints in `results/FORMAL_6SCENARIOS`, performs no optimizer update, and
stores the HyperComm evaluation rows below
`results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES`.
